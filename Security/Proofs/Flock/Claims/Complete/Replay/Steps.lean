import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.VBridge.Algebra.ZerocheckConverse

/-!
# Replaying the honest rep, one step at a time

The verifier replays a rep from the session record's rounds (`bindAndZerocheckWith`, …, `Flock.Transcript`, `Flock.Draw`, `Flock.Proof`, `Flock.Piop`, `Flock.Opening`, `Flock.Ligerito`, `Flock.Merkle`); the honest
prover (`repP` in the pure session `PS`, `sessionS`) wrote those rounds. Both absorb with the same actions, so their
pending bytes agree; where the prover sends the framed bytes as a round and records it (`squeezeP`), the verifier reads
that round back and compares its retained bytes exactly (`squeezeWith`), which calls no hash.

* **The prover's steps** in `PS`, from the transcript `t` and the bodies `l` sent so far: a transcript action
  (`liftTM_go`), a bind (`bind_go`), a squeeze (`squeezeP_go`: the round is recorded and its coin is the reply at
  `l.size`), and a loop whose every step yields (`forIn_go`).
* **The verifier's steps**: a squeeze whose round retained the framed bytes returns the round's coins (`squeezeWith_of`,
  `sampleWith_of`, `sampleVecWith_of`, `sampleKWith_of`).
* **The rounds agree** (`Agrees R P k₀`): from index `k₀` on, the verifier's rounds `R` carry the prover's rounds `P`
  (count, coins and retained bytes; the digest is not compared, as a retained round's squeeze never reads it).
-/

namespace FlockComplete.Replay

open Flock Transcript

/-! ## The pending bytes each absorb leaves (`label`, `absorbBytes`, `absorbSlice`, `absorbF`) -/

def pLabel (s : String) (P : ByteArray) : ByteArray := P ++ header 2 0 s.toUTF8.size ++ padded s.toUTF8

def pBytes (b P : ByteArray) : ByteArray := P ++ header 5 0 b.size ++ padded b

def pSlice (vs : Array F128) (P : ByteArray) : ByteArray := vs.foldl F128.pushBytes (P ++ header 3 2 vs.size)

def pF (v : F128) (P : ByteArray) : ByteArray := F128.pushBytes (P ++ header 3 1 1) v

/-- `f 0, …, f (n − 1)`, pushed in turn. -/
def pushes {α : Type} (f : ℕ → α) : ℕ → Array α
  | 0 => #[]
  | i + 1 => (pushes f i).push (f i)

theorem pushes_size {α : Type} (f : ℕ → α) (n : ℕ) : (pushes f n).size = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [pushes, ih]

theorem pushes_get {α : Type} [Inhabited α] (f : ℕ → α) (n : ℕ) : ∀ j < n, (pushes f n)[j]! = f j := by
  induction n with
  | zero => intro j hj; omega
  | succ n ih =>
    intro j hj
    rw [pushes, getElem!_pos _ j (by simp [pushes_size]; omega), Array.getElem_push]
    split
    · rw [← getElem!_pos]; exact ih j (by rename_i h; simpa [pushes_size] using h)
    · rename_i h; simp only [pushes_size] at h; congr 1; omega

theorem pushes_congr {α : Type} {f g : ℕ → α} {n : ℕ} (h : ∀ i < n, f i = g i) : pushes f n = pushes g n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [pushes, pushes, ih fun i hi => h i (by omega), h n (by omega)]

theorem pushes_eq_map {α : Type} (f : ℕ → α) (n : ℕ) : pushes f n = ((List.range n).map f).toArray := by
  induction n with
  | zero => rfl
  | succ n ih => rw [pushes, ih, List.range_succ, List.map_append]; simp

/-! ## The prover's steps in the pure session -/

theorem liftTM_go {α : Type} (x : TM α) (t : Transcript) (l : Array Claims.Msg) :
    ((liftTM x : TMW PS α).run t).go l = (x t, l) := rfl

theorem liftTM_ok {α : Type} {x : TM α} {t t' : Transcript} {a : α} (h : x t = .ok (a, t')) (l : Array Claims.Msg) :
    ((liftTM x : TMW PS α).run t).go l = (.ok (a, t'), l) := by
  rw [liftTM_go, h]

theorem bind_go {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    (h : (x.run t).go l = (.ok (a, t'), l')) : ((x >>= f).run t).go l = ((f a).run t').go l' := by
  rw [StateT.run_bind, go_bind, h]

theorem pure_go {α : Type} (a : α) (t : Transcript) (l : Array Claims.Msg) :
    ((pure a : TMW PS α).run t).go l = (.ok (a, t), l) := rfl

/-- The round a prover's squeeze of `n` coins of `kind` records from `t`, its coins `coin`'s first `n`. -/
def sent (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) : Round :=
  ⟨n, ByteArray.empty, (Array.range n).map coin, some (framed t kind n)⟩

/-- The prover's transcript after that squeeze. -/
def afterSq (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) : Transcript :=
  ⟨t.rounds.push (sent t kind n coin), t.k + 1, ByteArray.empty⟩

theorem squeezeP_go (resp : ℕ → Claims.Msg) (kind : UInt8) {n : ℕ} (hn : n ≠ 0) (t : Transcript)
    (l : Array Claims.Msg) :
    ((squeezeP (asksS resp) kind n).run t).go l =
      (.ok ((Array.range n).map (coinOf (resp l.size)), afterSq t kind n (coinOf (resp l.size))),
        l.push (Claims.toMsg (framed t kind n))) := by
  simp only [squeezeP, StateT.run, hn, ite_false]
  rfl

theorem sampleP_go (resp : ℕ → Claims.Msg) (t : Transcript) (l : Array Claims.Msg) :
    ((sampleP (asksS resp)).run t).go l =
      (.ok (coinOf (resp l.size) 0, afterSq t 1 1 (coinOf (resp l.size))), l.push (Claims.toMsg (framed t 1 1))) := by
  unfold sampleP
  rw [bind_go (squeezeP_go resp 1 one_ne_zero t l)]
  show (Except.ok (_, _), _) = _
  simp

theorem sampleVecP_go (resp : ℕ → Claims.Msg) {n : ℕ} (hn : n ≠ 0) (t : Transcript) (l : Array Claims.Msg) :
    ((sampleVecP (asksS resp) n).run t).go l =
      (.ok ((Array.range n).map (coinOf (resp l.size)), afterSq t 2 n (coinOf (resp l.size))),
        l.push (Claims.toMsg (framed t 2 n))) :=
  squeezeP_go resp 2 hn t l

theorem sampleKP_go (resp : ℕ → Claims.Msg) (t : Transcript) (l : Array Claims.Msg) :
    ((sampleKP (asksS resp)).run t).go l =
      (.ok (⟨coinOf (resp l.size) 0, coinOf (resp l.size) 1⟩, afterSq t 2 2 (coinOf (resp l.size))),
        l.push (Claims.toMsg (framed t 2 2))) := by
  unfold sampleKP
  rw [bind_go (squeezeP_go resp 2 two_ne_zero t l)]
  show (Except.ok (_, _), _) = _
  simp

/-- **A loop whose every step yields**, in the prover's monad: if step `i`, from the state `S i` and the transcript
`T i` with `L + i` bodies sent, yields `S (i + 1)` and `T (i + 1)` and sends `B i`, the loop over `[0:n]` from `S 0`
returns `S n` and `T n` and sends `B 0, …, B (n − 1)`. -/
theorem forIn_go {σ : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (S : ℕ → σ) (T : ℕ → Transcript)
    (B : ℕ → Claims.Msg) (L : ℕ)
    (hbody : ∀ i (l : Array Claims.Msg), l.size = L + i →
      ((body i (S i)).run (T i)).go l = (.ok (.yield (S (i + 1)), T (i + 1)), l.push (B i)))
    (n : ℕ) (l : Array Claims.Msg) (hl : l.size = L) :
    ((forIn [:n] (S 0) body).run (T 0)).go l = (.ok (S n, T n), l ++ ((List.range n).map B).toArray) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ k a (l : Array Claims.Msg), l.size = L + a →
      ((forIn (List.range' a k) (S a) body).run (T a)).go l =
        (.ok (S (a + k), T (a + k)), l ++ ((List.range' a k).map B).toArray) by
    simpa [List.range_eq_range'] using h n 0 l (by simpa using hl)
  intro k
  induction k with
  | zero =>
    intro a l _
    show (Except.ok (S a, T a), l) = _
    simp
  | succ k ih =>
    intro a l hl
    rw [List.range'_succ, List.forIn_cons, bind_go (hbody a l hl)]
    dsimp only
    rw [ih (a + 1) (l.push (B a)) (by simp [hl]; omega), show a + 1 + k = a + (k + 1) by omega]
    simp

/-! ## The verifier's steps on retained rounds -/

/-- **A squeeze of a retained round** (`squeezeWith`): if round `t.k` issued `n ≠ 0` coins and retained exactly the
framed bytes, the squeeze returns its coins, whatever the hashes. -/
theorem squeezeWith_of (hs : Hashes V) (kind : UInt8) {n : ℕ} (hn : n ≠ 0) {t : Transcript} {r : Round}
    (hr : t.rounds[t.k]? = some r) (hN : r.n = n) (hb : r.bytes = some (framed t kind n)) :
    squeezeWith hs kind n t = .ok (r.coins, ⟨t.rounds, t.k + 1, ByteArray.empty⟩) := by
  unfold squeezeWith Round.answersWith
  simp only [hn, ite_false, hr, hN, ne_eq, not_true_eq_false, hb]
  rfl

/-- The verifier's rounds `R` carry the prover's round `ρ` at `j`: its count, coins and retained bytes. -/
def Carries (R : Array Round) (j : ℕ) (ρ : Round) : Prop :=
  ∃ r, R[j]? = some r ∧ r.n = ρ.n ∧ r.coins = ρ.coins ∧ r.bytes = ρ.bytes

/-- **The rounds agree** from `k₀` on: the verifier's rounds `R` carry each of the prover's rounds `P` there. -/
def Agrees (R P : Array Round) (k₀ : ℕ) : Prop := ∀ j (h : j < P.size), k₀ ≤ j → Carries R j P[j]

/-- **A squeeze the prover sent** reads back its coins: at the verifier's transcript `⟨R, k, P⟩`, with the prover's
`t` at the same round and pending bytes and `R` carrying its round at `k`. -/
theorem squeezeWith_sent (hs : Hashes V) (kind : UInt8) {n : ℕ} (hn : n ≠ 0) {R : Array Round} {t : Transcript}
    {coin : ℕ → F128} (hc : Carries R t.k (sent t kind n coin)) :
    squeezeWith hs kind n ⟨R, t.k, t.pending⟩ = .ok ((Array.range n).map coin, ⟨R, t.k + 1, ByteArray.empty⟩) := by
  obtain ⟨r, hr, hN, hC, hb⟩ := hc
  rw [squeezeWith_of hs kind hn (t := ⟨R, t.k, t.pending⟩) (r := r) hr hN hb, hC]
  rfl

theorem sampleWith_sent (hs : Hashes V) {R : Array Round} {t : Transcript} {coin : ℕ → F128}
    (hc : Carries R t.k (sent t 1 1 coin)) :
    sampleWith hs ⟨R, t.k, t.pending⟩ = .ok (coin 0, ⟨R, t.k + 1, ByteArray.empty⟩) := by
  unfold sampleWith
  rw [Flock.RecAlgebra.bind_of_ok (squeezeWith_sent hs 1 one_ne_zero hc)]
  show Except.ok (_, _) = _
  simp

/-- `sampleWith_sent` at a transcript `u` given field by field, so that `u` unifies with any form of the state. -/
theorem sampleWith_sent_at (hs : Hashes V) {R : Array Round} {t : Transcript} {coin : ℕ → F128} (u : Transcript)
    (hR : u.rounds = R) (hk : u.k = t.k) (hp : u.pending = t.pending) (hc : Carries R t.k (sent t 1 1 coin)) :
    sampleWith hs u = .ok (coin 0, ⟨R, t.k + 1, ByteArray.empty⟩) := by
  obtain ⟨R', k, P⟩ := u
  dsimp only at hR hk hp
  subst hR hk hp
  exact sampleWith_sent hs hc

theorem sampleVecWith_sent (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {R : Array Round} {t : Transcript}
    {coin : ℕ → F128} (hc : Carries R t.k (sent t 2 n coin)) :
    sampleVecWith hs n ⟨R, t.k, t.pending⟩ = .ok ((Array.range n).map coin, ⟨R, t.k + 1, ByteArray.empty⟩) :=
  squeezeWith_sent hs 2 hn hc

/-- `sampleVecWith_sent` at a transcript `u` given field by field. -/
theorem sampleVecWith_sent_at (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {R : Array Round} {t : Transcript}
    {coin : ℕ → F128} (u : Transcript) (hR : u.rounds = R) (hk : u.k = t.k) (hp : u.pending = t.pending)
    (hc : Carries R t.k (sent t 2 n coin)) :
    sampleVecWith hs n u = .ok ((Array.range n).map coin, ⟨R, t.k + 1, ByteArray.empty⟩) := by
  obtain ⟨R', k, P⟩ := u
  dsimp only at hR hk hp
  subst hR hk hp
  exact sampleVecWith_sent hs hn hc

theorem sampleKWith_sent (hs : Hashes V) {R : Array Round} {t : Transcript} {coin : ℕ → F128}
    (hc : Carries R t.k (sent t 2 2 coin)) :
    sampleKWith hs ⟨R, t.k, t.pending⟩ = .ok (⟨coin 0, coin 1⟩, ⟨R, t.k + 1, ByteArray.empty⟩) := by
  unfold sampleKWith
  rw [Flock.RecAlgebra.bind_of_ok (squeezeWith_sent hs 2 two_ne_zero hc)]
  show Except.ok (_, _) = _
  simp

/-- `A`'s rounds are the first of `B`'s. -/
def Prefix (A B : Array Round) : Prop := ∃ C, B = A ++ C

theorem Prefix.refl (A : Array Round) : Prefix A A := ⟨#[], by simp⟩

theorem Prefix.trans {A B C : Array Round} (h₁ : Prefix A B) (h₂ : Prefix B C) : Prefix A C := by
  obtain ⟨D₁, rfl⟩ := h₁
  obtain ⟨D₂, rfl⟩ := h₂
  exact ⟨D₁ ++ D₂, by simp⟩

theorem Prefix.push (A : Array Round) (x : Round) : Prefix A (A.push x) := ⟨#[x], by simp⟩

/-- **The rounds agree at a pushed round**: where the prover's final rounds `B` begin with `A.push x`, from `k₀ ≤ |A|`,
the verifier's rounds carry `x` at `|A|`. -/
theorem Agrees.at_push {R B A : Array Round} {k₀ : ℕ} {x : Round} (h : Agrees R B k₀) (hp : Prefix (A.push x) B)
    (hk : k₀ ≤ A.size) : Carries R A.size x := by
  obtain ⟨C, rfl⟩ := hp
  have hlt : A.size < (A.push x ++ C).size := by simp; omega
  have e : (A.push x ++ C)[A.size] = x := by
    rw [Array.getElem_append_left (by simp)]
    simp
  exact e ▸ h A.size hlt hk

theorem Agrees.mono {R P : Array Round} {k₀ k₁ : ℕ} (h : Agrees R P k₀) (hk : k₀ ≤ k₁) : Agrees R P k₁ :=
  fun j hj hj₁ => h j hj (hk.trans hj₁)

/-- Rounds that carry the prover's final rounds carry every earlier stage's. -/
theorem Agrees.of_prefix {R P P' : Array Round} {k₀ : ℕ} (h : Agrees R P k₀) (hp : Prefix P' P) : Agrees R P' k₀ := by
  obtain ⟨C, rfl⟩ := hp
  intro j hj hk
  have := h j (by simp; omega) hk
  rwa [Array.getElem_append_left hj] at this

/-- **A squeeze of a retained round of one coin**, read through `t.rounds[t.k]!`. -/
theorem sampleWith_of (hs : Hashes V) {t : Transcript} (hN : (t.rounds[t.k]!).n = 1)
    (hb : (t.rounds[t.k]!).bytes = some (framed t 1 1)) :
    sampleWith hs t = .ok ((t.rounds[t.k]!).coins.getD 0 0, ⟨t.rounds, t.k + 1, ByteArray.empty⟩) := by
  have hlt : t.k < t.rounds.size := by
    by_contra hge
    rw [getElem!_neg t.rounds t.k hge] at hN
    cases hN
  have hr : t.rounds[t.k]? = some t.rounds[t.k]! := by
    rw [getElem!_pos t.rounds t.k hlt, Array.getElem?_eq_getElem hlt]
  unfold sampleWith
  rw [Flock.RecAlgebra.bind_of_ok (squeezeWith_of hs 1 one_ne_zero hr hN hb)]
  rfl

end FlockComplete.Replay
