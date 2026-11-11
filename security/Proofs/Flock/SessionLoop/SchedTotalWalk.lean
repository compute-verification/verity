module

public import Proofs.Flock.Claims.ZK.FrameRho
public import Proofs.Flock.Claims.Complete.Reps
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic

/-!
# The opening does not stop at admissible answers

`FrameAsks` counts the opening's asks and `FrameRho` follows its answers, both wherever it goes on, so both hold at a
program that stops. This file is their total counterpart: past its PIOP, the opening (`Finish.repOpening` on the live
channel `Drive.stepAsks coins true`), answered each round's count of coins within an admissibility `A`, never stops,
where `A` makes `ρ₁`'s answer independent of `ρ₀`'s.

* `ST A Q s hs`: from the answers `hs`, on answers in `A` of each round's count, `s` asks or ends in `Q`, and never
  stops (`st_bind`, `st_mono`); `st_feed`: fed answers in `A`, it is still `ST` where the feed leaves it.
* `totO A coins`: `ST` on the opening's channel past the coins it was given, as a `Total` (`totOT`, `totOL` over the
  transcript and Ligerito's prover state), so the walks of `Reps` show the opening's steps after `ρ₁` don't stop.
* `OS A P B p`, `OSL A P B p`: a transcript program over `OpenM`, and one over Ligerito's prover state, from a channel
  state and answers in `P`, on answers in `A`, does not stop and ends in `B`; with `bind`, `pure`, `if`, loops,
  transcript steps and squeezes (`os_*`, `osl_*`), the squeezes counting the answers (`PL`).
* **`ligeritoPZk_st`**, **`repOpening_st`**: the masked Ligerito, and the opening past the PIOP, don't stop where `A`
  makes the answer after `ρ₀`'s independent of it.
-/

@[expose] public section

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open _root_.Flock.Firewall.Finish (squeezeP sampleP sampleVecP sampleKP LP LigState liftT sendMsg powP challengeP
  oodIntroP queryP)
open FlockComplete.JsonRT (Total)
open FlockComplete.JsonRT.Total (bind' pure' lift_st forIn_range forIn_array)

/-- **Admissible answers**: `A hs cs`, the answer `cs` at the answers so far `hs`. -/
abbrev Adm := Hist → Array F128 → Prop

/-- **A step program that does not stop** at answers in `A`: from the answers `hs`, answered each round's count of
coins in `A`, it asks or ends, with a value and answers in `Q` where it ends. -/
def ST {α : Type} (A : Adm) (Q : α → Hist → Prop) : Step α → Hist → Prop
  | .done a, hs => Q a hs
  | .fail _, _ => False
  | .ask _ n k, hs => ∀ cs : Array F128, cs.size = n → A hs cs → ST A Q (k cs) (hs ++ [cs])

variable {A : Adm}

theorem st_bind {α β : Type} {Q : α → Hist → Prop} {R : β → Hist → Prop} {f : α → Step β}
    (hf : ∀ a hs, Q a hs → ST A R (f a) hs) : ∀ (s : Step α) (hs : Hist), ST A Q s hs → ST A R (s.bind f) hs
  | .done a, hs, h => hf a hs h
  | .fail _, _, h => h.elim
  | .ask _ _ k, hs, h => fun cs hn hc => st_bind hf (k cs) (hs ++ [cs]) (h cs hn hc)

theorem st_mono {α : Type} {Q Q' : α → Hist → Prop} (h : ∀ a hs, Q a hs → Q' a hs) :
    ∀ (s : Step α) (hs : Hist), ST A Q s hs → ST A Q' s hs
  | .done a, hs, hq => h a hs hq
  | .fail _, _, hq => hq.elim
  | .ask _ _ k, hs, hq => fun cs hn hc => st_mono h (k cs) (hs ++ [cs]) (hq cs hn hc)

theorem st_ne_fail {α : Type} {Q : α → Hist → Prop} {s : Step α} {hs : Hist} (h : ST A Q s hs) (e : String) :
    s ≠ .fail e := by
  rintro rfl
  exact h

theorem feed_done {α : Type} (a : α) (L : List (Array F128)) (rs : Array (ByteArray × ℕ)) :
    (Step.done a).feed L rs = (.done a, rs) := by
  cases L <;> rfl

theorem feed_fail {α : Type} (e : String) (L : List (Array F128)) (rs : Array (ByteArray × ℕ)) :
    (Step.fail e : Step α).feed L rs = (.fail e, rs) := by
  cases L <;> rfl

theorem feed_ask {α : Type} (c : ByteArray) (n : ℕ) (k : Array F128 → Step α) (x : Array F128)
    (L : List (Array F128)) (rs : Array (ByteArray × ℕ)) :
    (Step.ask c n k).feed (x :: L) rs = (k x).feed L (rs.push (c, n)) :=
  rfl

/-- **Fed, a program keeps the rounds it had sent**, and sends more. -/
theorem feed_keeps {α : Type} : ∀ (s : Step α) (L : List (Array F128)) (rs : Array (ByteArray × ℕ)),
    rs.size ≤ (s.feed L rs).2.size ∧ ∀ i < rs.size, (s.feed L rs).2[i]? = rs[i]?
  | s, [], rs => by rw [Loop.feed_nil]; exact ⟨le_rfl, fun _ _ => rfl⟩
  | .done a, _ :: _, rs => by rw [feed_done]; exact ⟨le_rfl, fun _ _ => rfl⟩
  | .fail e, _ :: _, rs => by rw [feed_fail]; exact ⟨le_rfl, fun _ _ => rfl⟩
  | .ask c n k, x :: L, rs => by
    rw [feed_ask]
    obtain ⟨hl, he⟩ := feed_keeps (k x) L (rs.push (c, n))
    rw [Array.size_push] at hl
    refine ⟨by omega, fun i hi => ?_⟩
    rw [he i (by rw [Array.size_push]; omega), Array.getElem?_push, ite_eq_right (Nat.ne_of_lt hi)]

/-- **Fed answers in `A`, a program that does not stop** does not stop where the feed leaves it, on the answers it took:
each answer fed of its round's count and in `A` at the answers before it, `rs`'s rounds and the feed's `j`-th the round
after them. -/
theorem st_feed {α : Type} {Q : α → Hist → Prop} : ∀ (s : Step α) (L : List (Array F128))
    (rs : Array (ByteArray × ℕ)) (hs : Hist), ST A Q s hs →
    (∀ j (hj : j < L.length) c n, (s.feed L rs).2[rs.size + j]? = some (c, n) →
      L[j].size = n ∧ A (hs ++ L.take j) L[j]) →
    ST A Q (s.feed L rs).1 (hs ++ L.take ((s.feed L rs).2.size - rs.size))
  | s, [], rs, hs, h, _ => by
    rw [Loop.feed_nil]
    simpa only [List.take_nil, List.append_nil] using h
  | .done a, _ :: _, rs, hs, h, _ => by
    rw [feed_done]
    simpa only [Nat.sub_self, List.take_zero, List.append_nil] using h
  | .fail e, _ :: _, rs, hs, h, _ => by
    rw [feed_fail]
    exact h.elim
  | .ask c n k, x :: L, rs, hs, h, hA => by
    rw [feed_ask] at hA ⊢
    have hl := (feed_keeps (k x) L (rs.push (c, n))).1
    rw [Array.size_push] at hl
    have hx : x.size = n ∧ A hs x := by
      have e := (feed_keeps (k x) L (rs.push (c, n))).2 rs.size (by rw [Array.size_push]; omega)
      rw [Array.getElem?_push_size] at e
      have := hA 0 (by simp only [List.length_cons]; omega) c n (by rw [Nat.add_zero]; exact e)
      simpa only [List.take_zero, List.append_nil, List.getElem_cons_zero] using this
    have ih := st_feed (k x) L (rs.push (c, n)) (hs ++ [x]) (h x hx.1 hx.2) fun j hj c' n' e => by
      have := hA (j + 1) (by simp only [List.length_cons]; omega) c' n'
        (by rw [← e, Array.size_push, Nat.add_assoc, Nat.add_comm 1 j])
      simpa only [List.take_succ_cons, List.getElem_cons_succ, List.append_assoc, List.singleton_append] using this
    rw [Array.size_push] at ih
    rw [show ((k x).feed L (rs.push (c, n))).2.size - rs.size =
        ((k x).feed L (rs.push (c, n))).2.size - (rs.size + 1) + 1 by omega, List.take_succ_cons]
    simpa only [List.append_assoc, List.singleton_append] using ih

/-! ## Not stopping as a `Total` -/

/-- **Not stopping on the opening's channel** past the coins it was given, at answers in `A`. -/
def totO (A : Adm) (coins : Got) : Total OpenM where
  T x P := ∀ got hs, Past coins got → ST A (fun r _ => P r.1 ∧ Past coins r.2) (x.run got) hs
  mono h hq got hs hp := st_mono (fun r _ hr => ⟨hq _ hr.1, hr.2⟩) _ _ (h got hs hp)
  pure _ h _ _ hp := ⟨h, hp⟩
  bind hx hf got hs hp := by
    rw [StateT.run_bind]
    exact st_bind (fun r hs' hr => hf r.1 hr.1 r.2 hs' hr.2) _ _ (hx got hs hp)
  lift v a hv h _ _ hp := by
    subst hv
    exact ⟨h, hp⟩

/-- Not stopping over the transcript. -/
abbrev totOT (A : Adm) (coins : Got) : Total (TMW OpenM) := (totO A coins).st Flock.Transcript

/-- Not stopping over Ligerito's state too. -/
abbrev totOL (A : Adm) (coins : Got) : Total (LP OpenM) := (totOT A coins).st LigState

variable {coins : Got}

theorem past_push {got : Got} (h : Past coins got) (cs : Array F128) : Past coins (got.push cs) := by
  unfold Past at h ⊢
  rw [Array.size_push]
  omega

/-- **A live round does not stop** past the coins it was given, at an answer of its count in `A`. -/
theorem st_round_live {Q : (ℕ → F128) × Got → Hist → Prop} (coins got : Got) (c : ByteArray) (hs : Hist)
    (h : Past coins got) (hQ : ∀ cs : Array F128, A hs cs → Q ((fun i => cs[i]!), got.push cs) (hs ++ [cs])) :
    ST A Q (((stepAsks coins true).round c).run got) hs := by
  have hn : coins[got.size]? = none := Array.getElem?_eq_none h
  simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind, hn,
    StateT.lift, ST]
  intro cs hc hA
  simp only [hc, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  exact hQ cs hA

theorem round_tot (coins : Got) (c : ByteArray) : (totO A coins).T ((stepAsks coins true).round c) fun _ => True :=
  fun got hs hp => st_round_live coins got c hs hp fun cs _ => ⟨trivial, past_push hp cs⟩

theorem squeezeP_tot (coins : Got) (kind : UInt8) (n : ℕ) :
    (totOT A coins).T (squeezeP (stepAsks coins true) kind n) fun _ => True := by
  intro t
  unfold squeezeP
  split
  · exact (totO A coins).pure' _
  · exact (totO A coins).bind' (round_tot coins _) fun _ => (totO A coins).pure' _

theorem sampleP_tot (coins : Got) : (totOT A coins).T (sampleP (stepAsks coins true)) fun _ => True :=
  (totOT A coins).bind' (squeezeP_tot coins 1 1) fun _ => (totOT A coins).pure' _

theorem sampleVecP_tot (coins : Got) (n : ℕ) :
    (totOT A coins).T (sampleVecP (stepAsks coins true) n) fun _ => True :=
  squeezeP_tot coins 2 n

theorem sampleKP_tot (coins : Got) : (totOT A coins).T (sampleKP (stepAsks coins true)) fun _ => True :=
  (totOT A coins).bind' (squeezeP_tot coins 2 2) fun _ => (totOT A coins).pure' _

theorem liftT_tot {α : Type} {x : TMW OpenM α} (h : (totOT A coins).T x fun _ => True) :
    (totOL A coins).T (liftT x) fun _ => True :=
  (totOT A coins).lift_st h

theorem sendMsg_tot (m : _root_.Flock.F256 × _root_.Flock.F256) :
    (totOL A coins).T (sendMsg (M := OpenM) m) fun _ => True := by
  unfold sendMsg
  repeat (first
    | (with_reducible apply bind')
    | (with_reducible intro _)
    | (with_reducible apply pure')
    | (with_reducible apply FlockComplete.JsonRT.Total.modify)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply FlockComplete.JsonRT.Total.liftTM; intro _; exact ⟨_, rfl⟩))

theorem powP_tot (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    (totOL A coins).T (powP (M := OpenM) nonce kind) fun _ => True := by
  unfold powP
  repeat (first
    | extract_lets
    | (with_reducible apply bind')
    | (with_reducible intro _)
    | (with_reducible apply FlockComplete.JsonRT.Total.get)
    | (with_reducible apply FlockComplete.JsonRT.Total.set)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply FlockComplete.JsonRT.Total.liftTM; intro _; exact ⟨_, rfl⟩))

theorem challengeP_tot (coins : Got) (split : Bool) :
    (totOL A coins).T (challengeP (stepAsks coins true) split) fun _ => True := by
  unfold challengeP
  repeat (first
    | extract_lets
    | (with_reducible apply bind')
    | (with_reducible intro _)
    | (with_reducible apply pure')
    | (with_reducible apply FlockComplete.JsonRT.Total.get)
    | (with_reducible apply FlockComplete.JsonRT.Total.modify)
    | (with_reducible apply sendMsg_tot)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply sampleKP_tot)
    | (with_reducible apply FlockComplete.JsonRT.Total.ite)
    | split)

theorem oodIntroP_tot (coins : Got) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    (totOL A coins).T (oodIntroP (stepAsks coins true) nonce g dim) fun _ => True := by
  unfold oodIntroP
  repeat (first
    | extract_lets
    | (with_reducible apply bind')
    | (with_reducible intro _)
    | (with_reducible apply FlockComplete.JsonRT.Total.modify)
    | (with_reducible apply sendMsg_tot)
    | (with_reducible apply powP_tot)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply sampleP_tot)
    | (with_reducible apply sampleVecP_tot)
    | (with_reducible apply FlockComplete.JsonRT.Total.liftTM; intro _; exact ⟨_, rfl⟩))

theorem queryP_tot (coins : Got) (nonce : ℕ → ℕ → UInt64) (ms : _root_.Flock.MerkleScheme) (lv : _root_.Flock.Level)
    (cm : _root_.Flock.Firewall.Finish.Commit) :
    (totOL A coins).T (queryP (stepAsks coins true) nonce ms lv cm) fun _ => True := by
  unfold queryP
  repeat (first
    | extract_lets
    | (with_reducible apply bind')
    | (with_reducible intro _)
    | (with_reducible apply pure')
    | (with_reducible apply powP_tot)
    | (with_reducible apply liftT_tot)
    | (with_reducible apply sampleVecP_tot))

/-! ## Transcript programs over `OpenM` that don't stop -/

/-- **A transcript program over `OpenM` that does not stop**: from a channel state and answers in `P`, on answers in
`A`, it does not stop, and where it ends it ends in `B`. -/
def OS {α : Type} (A : Adm) (P : Got → Hist → Prop) (B : α → Got → Hist → Prop) (p : TMW OpenM α) : Prop :=
  ∀ t got hs, P got hs → ST A (fun r hs' => B r.1.1 r.2 hs') ((p.run t).run got) hs

/-- Past the coins given, at `m` answers. -/
abbrev PL (coins : Got) (m : ℕ) : Got → Hist → Prop := fun g hs => Past coins g ∧ hs.length = m

theorem os_bind {α β : Type} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} {C : β → Got → Hist → Prop}
    {p : TMW OpenM α} {f : α → TMW OpenM β} (hp : OS A P B p) (hf : ∀ a, OS A (B a) C (f a)) : OS A P C (p >>= f) := by
  intro t got hs hP
  rw [StateT.run_bind, StateT.run_bind]
  exact st_bind (fun r hs' h => hf r.1.1 r.1.2 r.2 hs' h) _ _ (hp t got hs hP)

theorem os_mono {α : Type} {P P' : Got → Hist → Prop} {B B' : α → Got → Hist → Prop} {p : TMW OpenM α}
    (h : OS A P B p) (hP : ∀ g hs, P' g hs → P g hs) (hB : ∀ a g hs, B a g hs → B' a g hs) : OS A P' B' p :=
  fun t got hs h' => st_mono (fun _ _ hb => hB _ _ _ hb) _ _ (h t got hs (hP _ _ h'))

theorem os_pure {α : Type} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} (a : α)
    (h : ∀ g hs, P g hs → B a g hs) : OS A P B (pure a : TMW OpenM α) :=
  fun _ got hs hP => h got hs hP

/-- A step that neither asks nor stops nor moves the channel keeps `P`. -/
theorem os_det {α : Type} {P : Got → Hist → Prop} {p : TMW OpenM α}
    (h : ∀ t got, ∃ r, ((p.run t).run got) = .done (r, got)) : OS A P (fun _ => P) p := by
  intro t got hs hP
  obtain ⟨r, e⟩ := h t got
  rw [e]
  exact hP

theorem os_liftTM {α : Type} {P : Got → Hist → Prop} (x : Flock.Transcript.TM α) (hx : ∀ t, ∃ p, x t = .ok p) :
    OS A P (fun _ => P) (liftTM x : TMW OpenM α) :=
  os_det fun t got => by
    obtain ⟨p, hp⟩ := hx t
    refine ⟨p, ?_⟩
    show Step.bind (monadLift (x t)) _ = _
    rw [hp]
    rfl

/-- `p` doesn't stop past the coins given where it returns in `totOT`. -/
theorem os_of_tot {α : Type} {p : TMW OpenM α} {Q : α → Prop} (h : (totOT A coins).T p Q) :
    OS A (fun g _ => Past coins g) (fun a g _ => Q a ∧ Past coins g) p :=
  fun t got hs hP => h t got hs hP

/-- **A live squeeze's answer**: of `n ≠ 0` coins, it takes one answer in `A`, its value the answer's words. -/
theorem os_squeezeP_live (kind : UInt8) {n : ℕ} (hn0 : n ≠ 0) (h0 : Hist) :
    OS A (fun g hs => Past coins g ∧ hs = h0)
      (fun v g hs => Past coins g ∧ ∃ cs, hs = h0 ++ [cs] ∧ A h0 cs ∧ v = (Array.range n).map fun i => cs[i]!)
      (squeezeP (stepAsks coins true) kind n) := by
  rintro t got hs ⟨hI, rfl⟩
  simp only [squeezeP, StateT.run, hn0, ↓reduceIte, bind, StateT.bind]
  refine st_bind (Q := fun r hs' => ∃ cs, r = ((fun i => cs[i]!), got.push cs) ∧ hs' = hs ++ [cs] ∧ A hs cs)
    (fun r hs' h => ?_) _ _ (st_round_live coins got _ hs hI fun cs hA => ⟨cs, rfl, rfl, hA⟩)
  obtain ⟨cs, rfl, rfl, hA⟩ := h
  exact ⟨past_push hI cs, cs, rfl, hA, rfl⟩

/-- **A live squeeze's count**: a squeeze of `n` coins takes one answer, none for `n = 0`. -/
theorem os_squeezeP (kind : UInt8) (n m : ℕ) :
    OS A (PL coins m) (fun _ => PL coins (m + (sq n).length)) (squeezeP (stepAsks coins true) kind n) := by
  rintro t got hs ⟨hI, hl⟩
  by_cases h0 : n = 0
  · subst h0
    exact ⟨hI, by simp [sq, hl]⟩
  · refine st_mono ?_ _ _ (os_squeezeP_live kind h0 hs t got hs ⟨hI, rfl⟩)
    rintro r hs' ⟨hP, cs, rfl, -, -⟩
    exact ⟨hP, by simp [sq, h0, hl]⟩

theorem os_sampleP (m : ℕ) : OS A (PL coins m) (fun _ => PL coins (m + 1)) (sampleP (stepAsks coins true)) :=
  os_bind (os_squeezeP 1 1 m) fun _ => os_pure _ fun _ _ h => h

theorem os_sampleVecP (n m : ℕ) :
    OS A (PL coins m) (fun _ => PL coins (m + (sq n).length)) (sampleVecP (stepAsks coins true) n) :=
  os_squeezeP 2 n m

theorem os_sampleKP (m : ℕ) : OS A (PL coins m) (fun _ => PL coins (m + 1)) (sampleKP (stepAsks coins true)) :=
  os_bind (os_squeezeP 2 2 m) fun _ => os_pure _ fun _ _ h => h

/-- **A live `F256` squeeze's value is its answer's first two words**, its answer in `A`. -/
theorem os_sampleKP_val (h0 : Hist) :
    OS A (fun g hs => Past coins g ∧ hs = h0)
      (fun v g hs => Past coins g ∧ ∃ cs, hs = h0 ++ [cs] ∧ A h0 cs ∧ v = (⟨cs[0]!, cs[1]!⟩ : _root_.Flock.F256))
      (sampleKP (stepAsks coins true)) := by
  unfold sampleKP
  refine os_bind (os_squeezeP_live 2 (by decide) h0) fun v => ?_
  refine os_pure _ ?_
  rintro g hs ⟨hI, cs, rfl, hA, rfl⟩
  exact ⟨hI, cs, rfl, hA, by simp⟩

/-- **A loop over `[0:n]`**, step `i` from `J i` yielding into `J (i + 1)`. -/
theorem os_forIn {σ : Type} (n : ℕ) (J : ℕ → σ → Got → Hist → Prop) (body : ℕ → σ → TMW OpenM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, OS A (J i u) (fun r g hs => ∃ v, r = .yield v ∧ J (i + 1) v g hs) (body i u)) (u : σ) :
    OS A (J 0 u) (fun r => J n r) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → OS A (J a u) (fun r => J n r) (forIn (List.range' a k) u body) from
    key n 0 u (by omega)
  intro k
  induction k with
  | zero =>
    intro a u ha
    have e : a = n := by omega
    subst e
    exact os_pure u fun _ _ h => h
  | succ k ih =>
    intro a u ha
    rw [List.range'_succ, List.forIn_cons]
    refine os_bind (hb a (by omega) u) fun r => ?_
    intro t got hs ⟨v, hv, hJ⟩
    subst hv
    exact ih (a + 1) v (by omega) t got hs hJ

theorem os_forIn_bind {σ β : Type} {n : ℕ} (J : ℕ → σ → Got → Hist → Prop) {P : Got → Hist → Prop}
    {C : β → Got → Hist → Prop} {body : ℕ → σ → TMW OpenM (ForInStep σ)} {u : σ} {f : σ → TMW OpenM β}
    (h0 : ∀ g hs, P g hs → J 0 u g hs)
    (hb : ∀ i < n, ∀ u, OS A (J i u) (fun r g hs => ∃ v, r = .yield v ∧ J (i + 1) v g hs) (body i u))
    (hf : ∀ r, OS A (J n r) C (f r)) : OS A P C (forIn [:n] u body >>= f) :=
  os_bind (os_mono (os_forIn n J body hb u) h0 fun _ _ _ h => h) hf

theorem st_run' {α : Type} {p : TMW OpenM α} {t : Flock.Transcript} {s : Got} {R : Hist → Prop} {hs : Hist}
    (h : ST A (fun _ hs' => R hs') ((p.run t).run s) hs) : ST A (fun _ hs' => R hs') ((p.run' t).run' s) hs := by
  refine st_bind (Q := fun _ hs' => R hs') (fun _ _ h => ?_) _ _
    (st_bind (Q := fun _ hs' => R hs') (fun _ _ h => ?_) _ _ h) <;> exact h

/-! ## Over Ligerito's prover state -/

/-- `OS` over the Ligerito prover's state: from any state. -/
def OSL {α : Type} (A : Adm) (P : Got → Hist → Prop) (B : α → Got → Hist → Prop) (p : LP OpenM α) : Prop :=
  ∀ st, OS A P (fun r => B r.1) (p.run st)

theorem osl_bind {α β : Type} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} {C : β → Got → Hist → Prop}
    {p : LP OpenM α} {f : α → LP OpenM β} (hp : OSL A P B p) (hf : ∀ a, OSL A (B a) C (f a)) :
    OSL A P C (p >>= f) := fun st => by
  rw [StateT.run_bind]
  exact os_bind (hp st) fun q => hf q.1 q.2

theorem osl_mono {α : Type} {P P' : Got → Hist → Prop} {B B' : α → Got → Hist → Prop} {p : LP OpenM α}
    (h : OSL A P B p) (hP : ∀ g hs, P' g hs → P g hs) (hB : ∀ a g hs, B a g hs → B' a g hs) : OSL A P' B' p :=
  fun st => os_mono (h st) hP fun _ _ _ hb => hB _ _ _ hb

theorem osl_pure {α : Type} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} (a : α)
    (h : ∀ g hs, P g hs → B a g hs) : OSL A P B (pure a : LP OpenM α) := fun st => by
  rw [StateT.run_pure]
  exact os_pure (B := fun r => B r.1) (a, st) h

theorem osl_ite_same {α : Type} (c : Prop) [Decidable c] {P : Got → Hist → Prop} {B : α → Got → Hist → Prop}
    {p q : LP OpenM α} (h1 : c → OSL A P B p) (h2 : ¬ c → OSL A P B q) : OSL A P B (if c then p else q) := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact h1 hc
  · rw [ite_eq_right hc]; exact h2 hc

theorem osl_liftT {α : Type} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} {m : TMW OpenM α}
    (h : OS A P B m) : OSL A P B (liftT m : LP OpenM α) := fun st => by
  show OS A _ _ (StateT.lift m st)
  exact os_bind h fun a => os_pure (B := fun (r : α × LigState) => B r.1) (a, st) fun _ _ hb => hb

/-- A step that neither asks nor stops nor moves the channel keeps `P`. -/
theorem osl_det {α : Type} {P : Got → Hist → Prop} {p : LP OpenM α}
    (h : ∀ st t got, ∃ r, (((p.run st).run t).run got) = .done (r, got)) : OSL A P (fun _ => P) p :=
  fun st => os_det (h st)

theorem osl_liftTM {α : Type} {P : Got → Hist → Prop} (x : Flock.Transcript.TM α) (hx : ∀ t, ∃ p, x t = .ok p) :
    OSL A P (fun _ => P) (liftT (liftTM x : TMW OpenM α)) :=
  osl_liftT (os_liftTM x hx)

theorem osl_sendMsg {P : Got → Hist → Prop} (m : _root_.Flock.F256 × _root_.Flock.F256) :
    OSL A P (fun _ => P) (sendMsg m : LP OpenM Unit) := by
  unfold sendMsg
  exact osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ =>
    osl_bind (osl_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => osl_liftTM _ fun _ => ⟨_, rfl⟩

theorem osl_powP {P : Got → Hist → Prop} (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    OSL A P (fun _ => P) (powP nonce kind : LP OpenM Unit) := by
  unfold powP
  exact osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ =>
    osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ => osl_liftTM _ fun _ => ⟨_, rfl⟩

/-- `p` doesn't stop past the coins given where it returns in `totOL`. -/
theorem osl_of_tot {α : Type} {p : LP OpenM α} {Q : α → Prop} (h : (totOL A coins).T p Q) :
    OSL A (fun g _ => Past coins g) (fun a g _ => Q a ∧ Past coins g) p :=
  fun st => os_of_tot (h st)

/-- **A loop over `[0:n]`**, step `i` from `J i` yielding into `J (i + 1)`. -/
theorem osl_forIn {σ : Type} (n : ℕ) (J : ℕ → σ → Got → Hist → Prop) (body : ℕ → σ → LP OpenM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, OSL A (J i u) (fun r g hs => ∃ v, r = .yield v ∧ J (i + 1) v g hs) (body i u)) (u : σ) :
    OSL A (J 0 u) (fun r => J n r) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → OSL A (J a u) (fun r => J n r) (forIn (List.range' a k) u body) from
    key n 0 u (by omega)
  intro k
  induction k with
  | zero =>
    intro a u ha
    have e : a = n := by omega
    subst e
    exact osl_pure u fun _ _ h => h
  | succ k ih =>
    intro a u ha
    rw [List.range'_succ, List.forIn_cons]
    refine osl_bind (hb a (by omega) u) fun r => ?_
    intro st t got hs ⟨v, hv, hJ⟩
    subst hv
    exact ih (a + 1) v (by omega) st t got hs hJ

theorem osl_forIn_bind {σ β : Type} {n : ℕ} (J : ℕ → σ → Got → Hist → Prop) {P : Got → Hist → Prop}
    {C : β → Got → Hist → Prop} {body : ℕ → σ → LP OpenM (ForInStep σ)} {u : σ} {f : σ → LP OpenM β}
    (h0 : ∀ g hs, P g hs → J 0 u g hs)
    (hb : ∀ i < n, ∀ u, OSL A (J i u) (fun r g hs => ∃ v, r = .yield v ∧ J (i + 1) v g hs) (body i u))
    (hf : ∀ r, OSL A (J n r) C (f r)) : OSL A P C (forIn [:n] u body >>= f) :=
  osl_bind (osl_mono (osl_forIn n J body hb u) h0 fun _ _ _ h => h) hf

/-- The answers so far, as a ghost. -/
theorem osl_ghost {α : Type} {I : Got → Prop} {R : Hist → Prop} {B : α → Got → Hist → Prop} {p : LP OpenM α}
    (h : ∀ h0, R h0 → OSL A (fun g hs => I g ∧ hs = h0) B p) : OSL A (fun g hs => I g ∧ R hs) B p :=
  fun st t got hs ⟨hI, hR⟩ => h hs hR st t got hs ⟨hI, rfl⟩

theorem osl_exists {α : Type} {I : Got → Prop} {Px : Array F128 → Prop} {h0 : Hist} {B : α → Got → Hist → Prop}
    {p : LP OpenM α} (h : ∀ cs, Px cs → OSL A (fun g hs => I g ∧ hs = h0 ++ [cs]) B p) :
    OSL A (fun g hs => I g ∧ ∃ cs, hs = h0 ++ [cs] ∧ Px cs) B p :=
  fun st t got hs ⟨hI, cs, he, hx⟩ => h cs hx st t got hs ⟨hI, he⟩

theorem os_run_bind {α β : Type} {P : Got → Hist → Prop} {B : α → Got → Hist → Prop} {C : β → Got → Hist → Prop}
    {p : LP OpenM α} {st : LigState} {f : α × LigState → TMW OpenM β} (hp : OSL A P B p)
    (hf : ∀ a, OS A (B a.1) C (f a)) : OS A P C (p.run st >>= f) :=
  os_bind (hp st) hf

/-! ## The opening -/

theorem sq_length {n : ℕ} (h : n ≠ 0) : (sq n).length = 1 := by
  simp [sq, h]

/-- **The ring switch's answers**: one a claim, then the batching's (none for no claims). -/
theorem ringSwitchP_st (sv : ℕ → Array F128) (claims : Array _root_.Flock.Claim) (m : ℕ) :
    OS A (PL coins m) (fun _ => PL coins (m + claims.size + (sq claims.size).length))
      (_root_.Flock.Firewall.Finish.ringSwitchP (stepAsks coins true) sv claims) := by
  unfold _root_.Flock.Firewall.Finish.ringSwitchP
  refine os_bind (os_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine os_forIn_bind (fun i _ => PL coins (m + i)) (fun _ _ h => h) (fun i _ u => ?_) fun r => ?_
  · refine os_bind (os_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
    refine os_bind (os_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
    refine os_bind (os_sampleVecP 7 _) fun _ => ?_
    refine os_pure _ fun _ _ h => ⟨_, rfl, h.1, ?_⟩
    have h2 := h.2
    have := sq_length (n := 7) (by decide)
    omega
  · refine os_bind (os_sampleVecP _ _) fun _ => ?_
    exact os_pure _ fun _ _ h => h

/-- **The masked Ligerito does not stop** from `m` answers, where `A` makes the answer after `ρ₀`'s, at `z`'s, `β₀`'s
and the `k₀` lanes' past `m`, independent of it. -/
theorem ligeritoPZk_st (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : F128)
    (cm0 : _root_.Flock.Firewall.Finish.Commit) (t b : Array F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (m : ℕ)
    (hρ : ∀ hs cs, hs.length = m + (sq sch.n).length + 1 + sch.k0 + 1 → A hs cs →
      _root_.Flock.Zk.rhoIndependent (⟨(hs.getD (m + (sq sch.n).length + 1 + sch.k0) #[])[0]!,
        (hs.getD (m + (sq sch.n).length + 1 + sch.k0) #[])[1]!⟩, ⟨cs[0]!, cs[1]!⟩) = true) :
    OS A (PL coins m) (fun _ g _ => Past coins g)
      (_root_.Flock.Firewall.Finish.ligeritoPZk (stepAsks coins true)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) ms sch T cm0 t b lay h rep tPad extra
        lanePad salt nonce) := by
  unfold _root_.Flock.Firewall.Finish.ligeritoPZk
  refine os_bind (os_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine os_bind (os_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine os_bind (os_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine os_bind (os_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  dsimp only
  refine os_run_bind (B := fun _ g _ => Past coins g) ?_ fun a => ?_
  rotate_left
  · obtain ⟨⟨_, _, _, _, _, _, _, _, _⟩, _⟩ := a
    exact os_pure _ fun _ _ h => h
  refine osl_bind (osl_liftT (os_sampleVecP _ m)) fun _ => ?_
  refine osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ => ?_
  refine osl_bind (osl_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
  refine osl_bind (osl_powP nonce 2) fun _ => ?_
  refine osl_bind (osl_liftT (os_sampleP _)) fun _ => ?_
  refine osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ => ?_
  refine osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ => ?_
  refine osl_bind (osl_sendMsg _) fun _ => ?_
  refine osl_forIn_bind (fun j _ g hs => (j < sch.k0 → PL coins (m + (sq sch.n).length + 1 + j) g hs) ∧
      (sch.k0 ≤ j → Past coins g)) (fun _ _ hA => ⟨fun _ => hA, fun _ => hA.1⟩) (fun i hi u => ?_) fun r => ?_
  · refine osl_mono (P := PL coins (m + (sq sch.n).length + 1 + i)) ?_ (fun _ _ hJ => hJ.1 hi) fun _ _ _ h => h
    refine osl_bind (osl_liftT (os_sampleKP _)) fun _ => ?_
    refine osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ => ?_
    apply osl_ite_same
    · intro hl
      have hi1 : i + 1 = sch.k0 := by simpa using hl
      refine osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ => ?_
      refine osl_bind (osl_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine osl_bind (osl_liftTM _ fun _ => ⟨_, rfl⟩) fun _ => ?_
      refine osl_mono (P := fun g hs => Past coins g ∧ hs.length = m + (sq sch.n).length + 1 + sch.k0) ?_
        (fun _ hs h => ⟨h.1, by have h2 := h.2; omega⟩) fun _ _ _ h => h
      apply osl_ghost
      intro h0 hl0
      refine osl_bind (osl_liftT (os_sampleKP_val h0)) fun rh0 => ?_
      apply osl_exists
      rintro cs0 ⟨-, hr0⟩
      refine osl_bind (osl_liftT (os_sampleKP_val (h0 ++ [cs0]))) fun rh1 => ?_
      apply osl_exists
      rintro cs1 ⟨hA1, hr1⟩
      have hind : _root_.Flock.Zk.rhoIndependent (rh0, rh1) = true := by
        have e0 : (h0 ++ [cs0]).getD (m + (sq sch.n).length + 1 + sch.k0) #[] = cs0 := by
          rw [← hl0, List.getD_eq_getElem?_getD, List.getElem?_append_right le_rfl, Nat.sub_self]
          simp
        have := hρ (h0 ++ [cs0]) cs1 (by simp [hl0]) hA1
        rw [e0] at this
        rw [hr0, hr1]
        exact this
      apply osl_ite_same
      · intro hc
        simp [hind] at hc
      · intro _
        refine osl_mono (osl_of_tot (Q := fun r => ∃ v, r = .yield v) ?_) (fun _ _ h => h.1)
          fun _ _ _ ⟨⟨v, hv⟩, hP⟩ => ⟨v, hv, fun h' => absurd h' (by omega), fun _ => hP⟩
        repeat (first
          | extract_lets
          | (with_reducible apply bind')
          | (with_reducible intro _)
          | (with_reducible apply pure')
          | (exact FlockComplete.JsonRT.Total.pure _ _ ⟨_, rfl⟩)
          | (with_reducible apply FlockComplete.JsonRT.Total.get)
          | (with_reducible apply FlockComplete.JsonRT.Total.modify)
          | (with_reducible apply sendMsg_tot)
          | (with_reducible apply liftT_tot)
          | (with_reducible apply FlockComplete.JsonRT.Total.liftTM; intro _; exact ⟨_, rfl⟩))
    · intro hl
      have hi1 : i + 1 ≠ sch.k0 := by simpa using hl
      refine osl_bind (osl_det fun _ _ _ => ⟨_, rfl⟩) fun _ => ?_
      refine osl_bind (osl_sendMsg _) fun _ => ?_
      exact osl_pure _ fun _ _ ⟨hI, hl'⟩ => ⟨_, rfl, fun _ => ⟨hI, by omega⟩, fun h' => absurd h' (by omega)⟩
  · refine osl_mono (osl_of_tot (Q := fun _ => True) ?_) (fun _ _ hJ => hJ.2 le_rfl) fun _ _ _ h => h.2
    repeat (first
      | extract_lets
      | (with_reducible apply bind')
      | (with_reducible intro _)
      | (with_reducible apply pure')
      | (with_reducible apply FlockComplete.JsonRT.Total.get)
      | (with_reducible apply FlockComplete.JsonRT.Total.modify)
      | (with_reducible apply sendMsg_tot)
      | (with_reducible apply powP_tot)
      | (with_reducible apply challengeP_tot)
      | (with_reducible apply oodIntroP_tot)
      | (with_reducible apply queryP_tot)
      | (with_reducible apply liftT_tot)
      | (with_reducible apply lift_st)
      | (with_reducible apply sampleP_tot)
      | (with_reducible apply sampleVecP_tot)
      | (with_reducible apply sampleKP_tot)
      | (with_reducible apply forIn_range)
      | (with_reducible apply forIn_array)
      | (with_reducible apply FlockComplete.JsonRT.Total.ite)
      | (with_reducible apply FlockComplete.JsonRT.Total.liftTM; intro _; exact ⟨_, rfl⟩)
      | split)

/-- **The opening past its PIOP does not stop** where `A` makes the answer after `ρ₀`'s, past the ring switch's, `z`'s,
`β₀`'s and the `k₀` lanes', independent of it. -/
theorem repOpening_st (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (lay : _root_.Flock.Zk.PadLayout)
    (tPad rep : ℕ) (g : Array F128) (cm0 : _root_.Flock.Firewall.Finish.Commit) (claims : Array _root_.Flock.Claim)
    (h : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (hρ : ∀ hs cs, hs.length = claims.size + (sq claims.size).length + (sq sch.n).length + 1 + sch.k0 + 1 → A hs cs →
      _root_.Flock.Zk.rhoIndependent
        (⟨(hs.getD (claims.size + (sq claims.size).length + (sq sch.n).length + 1 + sch.k0) #[])[0]!,
          (hs.getD (claims.size + (sq claims.size).length + (sq sch.n).length + 1 + sch.k0) #[])[1]!⟩,
          ⟨cs[0]!, cs[1]!⟩) = true) :
    OS A (PL coins 0) (fun _ g _ => Past coins g)
      (_root_.Flock.Firewall.Finish.repOpening (stepAsks coins true)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra
        lanePad salt) := by
  unfold _root_.Flock.Firewall.Finish.repOpening
  refine os_bind (os_mono (ringSwitchP_st _ claims 0) (fun _ _ h => h)
    (B' := fun _ => PL coins (claims.size + (sq claims.size).length))
    fun _ _ _ h => ⟨h.1, by have h2 := h.2; omega⟩) fun r => ?_
  obtain ⟨ss, T, sws⟩ := r
  dsimp only
  refine os_bind (ligeritoPZk_st _ sch _ _ _ _ _ _ _ _ _ _ _ _ _ hρ) fun r => ?_
  obtain ⟨_, _, _⟩ := r
  exact os_pure _ fun _ _ h => h

end Claims.ZKServer.Count
