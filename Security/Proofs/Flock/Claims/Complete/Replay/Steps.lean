import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.VBridge.Algebra.ZerocheckConverse

/-!
# Replaying the honest rep, one step at a time

The verifier replays a rep from the session record's rounds (`bindAndZerocheckWith`, …, PROTOCOL.md §6–§14); the honest
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

theorem sampleVecWith_sent (hs : Hashes V) {n : ℕ} (hn : n ≠ 0) {R : Array Round} {t : Transcript}
    {coin : ℕ → F128} (hc : Carries R t.k (sent t 2 n coin)) :
    sampleVecWith hs n ⟨R, t.k, t.pending⟩ = .ok ((Array.range n).map coin, ⟨R, t.k + 1, ByteArray.empty⟩) :=
  squeezeWith_sent hs 2 hn hc

theorem sampleKWith_sent (hs : Hashes V) {R : Array Round} {t : Transcript} {coin : ℕ → F128}
    (hc : Carries R t.k (sent t 2 2 coin)) :
    sampleKWith hs ⟨R, t.k, t.pending⟩ = .ok (⟨coin 0, coin 1⟩, ⟨R, t.k + 1, ByteArray.empty⟩) := by
  unfold sampleKWith
  rw [Flock.RecAlgebra.bind_of_ok (squeezeWith_sent hs 2 two_ne_zero hc)]
  show Except.ok (_, _) = _
  simp

/-- `Agrees` at a round the prover pushed. -/
theorem Agrees.carries {R P : Array Round} {k₀ : ℕ} (h : Agrees R P k₀) {j : ℕ} (hj : j < P.size) (hk : k₀ ≤ j) :
    Carries R j P[j] := h j hj hk

end FlockComplete.Replay
