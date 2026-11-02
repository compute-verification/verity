import Proofs.Flock.Claims.Complete.Replay.LigeritoRep
import Proofs.Flock.Claims.Complete.Json.Goals

/-!
# A returning rep's rounds are few and small

`repP_repOk`: a rep that returns (`repP`) from the empty transcript records a round, each of at most `65536` coins
(`RepOk`), at a table of `14 ≤ m ≤ 35` with `fast100`'s schedule and at most `65534` extra claims. Its transcript is
`Lg.out`'s after the ring switch's, lincheck's and zerocheck's (`repP_ligerito`), and each squeeze of those models is of
a few coins: zerocheck's `6`, `m − 13` and `1`; lincheck's `1`; the ring switch's `7` and one per claim; Ligerito's `2`,
`1`, the schedule's `n` and split dimensions (at most `n + 1`), and each level's `q` and `⌊log₂(q − 1)⌋ + 1`.

* `Bnd B N t`: `t` holds at least `N` rounds, each of at most `B` coins; kept by every absorb (`Lg.ap`) and by a squeeze
  of at most `B` coins (`Bnd.afterSq`).
-/

namespace FlockComplete.Replay

open Flock Transcript

/-- `t` holds at least `N` rounds, each of at most `B` coins. -/
def Bnd (B N : ℕ) (t : Transcript) : Prop := N ≤ t.rounds.size ∧ ∀ x ∈ t.rounds, x.n ≤ B

namespace Bnd

variable {B N : ℕ}

theorem afterSq {t : Transcript} (h : Bnd B N t) {n : ℕ} (hn : n ≤ B) (kind : UInt8) (coin : ℕ → F128) :
    Bnd B N (Replay.afterSq t kind n coin) := by
  refine ⟨by simp only [Replay.afterSq, Array.size_push]; exact Nat.le_succ_of_le h.1, fun x hx => ?_⟩
  simp only [Replay.afterSq, Array.mem_push] at hx
  rcases hx with hx | rfl
  · exact h.2 x hx
  · exact hn

theorem ap {t : Transcript} (h : Bnd B N t) (f : ByteArray → ByteArray) : Bnd B N (Lg.ap f t) := h

theorem mono {t : Transcript} (h : Bnd B N t) {N' : ℕ} (hN : N' ≤ N) : Bnd B N' t := ⟨hN.trans h.1, h.2⟩

theorem of_rounds {t t' : Transcript} (h : Bnd B N t) (hr : t'.rounds = t.rounds) : Bnd B N t' := by
  unfold Bnd
  rw [hr]
  exact h

/-- A squeeze from `t`'s rounds, at any cursor and absorbed bytes. -/
theorem sq {t : Transcript} (h : Bnd B N t) {n : ℕ} (hn : n ≤ B) (k : ℕ) (P : ByteArray) (kind : UInt8)
    (coin : ℕ → F128) : Bnd B N (Replay.afterSq ⟨t.rounds, k, P⟩ kind n coin) :=
  Bnd.afterSq (t := ⟨t.rounds, k, P⟩) (of_rounds h rfl) hn kind coin

end Bnd

namespace Zc

theorem bnd_loopT {B N m : ℕ} {cs : Challenges} {tr : Trace} {t₀ : Transcript} (h : Bnd B N t₀) (h1 : 1 ≤ B) :
    ∀ i, Bnd B N (loopT m cs tr t₀ i)
  | 0 => h
  | i + 1 => by
    simp only [loopT]
    exact Bnd.sq (bnd_loopT h h1 i) h1 _ _ _ _

/-- Zerocheck's squeezes are of `6`, `m − 13` and `1` coins. -/
theorem bnd_tEnd {B N m : ℕ} {digest cap : ByteArray} {cs : Challenges} {tr : Trace} {t : Transcript} (h : Bnd B N t)
    (h6 : 6 ≤ B) (hm : m - 13 ≤ B) : Bnd B N (tEnd m digest cap cs tr t) := by
  have h₂ : Bnd B N (t₂ digest cap cs t) := Bnd.sq h h6 _ _ _ _
  have h₃ : Bnd B N (t₃ m digest cap cs t) := Bnd.afterSq h₂ hm _ _
  have h₅ : Bnd B N (t₅ m digest cap cs tr t) := Bnd.sq h₃ (by omega) _ _ _ _
  exact Bnd.of_rounds (bnd_loopT h₅ (by omega) (m - 6)) (tEnd_rounds t)

end Zc

namespace Lc

theorem bnd_loopT {B N : ℕ} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Challenges}
    {t : Transcript} (h : Bnd B N t) (h1 : 1 ≤ B) : ∀ i, Bnd B N (loopT c m tr zc full cs t i)
  | 0 => by
    have h₂ : Bnd B N (t₂ cs t) := Bnd.sq h h1 _ _ _ _
    exact Bnd.afterSq h₂ h1 _ _
  | i + 1 => by
    simp only [loopT]
    exact Bnd.sq (bnd_loopT h h1 i) h1 _ _ _ _

/-- Lincheck's squeezes are of one coin. -/
theorem bnd_tEnd {B N : ℕ} {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Challenges}
    {t : Transcript} (h : Bnd B N t) (h1 : 1 ≤ B) : Bnd B N (tEnd c m tr zc full cs t) :=
  Bnd.sq (bnd_loopT h h1 _) h1 _ _ _ _

end Lc

namespace Rs

theorem bnd_loopT {B N : ℕ} {tr : Trace} {claims : Array Claim} {cs : Challenges} {t : Transcript} (h : Bnd B N t)
    (h7 : 7 ≤ B) : ∀ k, Bnd B N (loopT tr claims cs t k)
  | 0 => Bnd.of_rounds h rfl
  | k + 1 => Bnd.sq (bnd_loopT h h7 k) h7 _ _ _ _

/-- The ring switch's squeezes are of `7` coins and one per claim. -/
theorem bnd_tEnd {B N : ℕ} {tr : Trace} {claims : Array Claim} {cs : Challenges} {t : Transcript} (h : Bnd B N t)
    (h7 : 7 ≤ B) (hc : claims.size ≤ B) : Bnd B N (tEnd tr claims cs t) :=
  Bnd.afterSq (bnd_loopT h h7 _) hc _ _

end Rs

namespace Lg

theorem bnd_chal {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {st : LState} {t : Transcript} (h : Bnd B N t)
    (h2 : 2 ≤ B) : Bnd B N (chal p cf st t).2.2 :=
  (h.afterSq h2 _ _).ap _

theorem bnd_chals {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {acc : Array F256} {st : LState}
    {t : Transcript} (h : Bnd B N t) (h2 : 2 ≤ B) : ∀ j, Bnd B N (chals p cf acc st t j).2.2
  | 0 => h
  | j + 1 => bnd_chal (bnd_chals h h2 j) h2

theorem bnd_ood {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {dim split : ℕ} {st : LState} {t : Transcript}
    (h : Bnd B N t) (hd : dim ≤ B) (h1 : 1 ≤ B) : Bnd B N (ood p cf dim split st t).2 :=
  Bnd.afterSq ((((h.afterSq hd _ _).ap _).ap _).ap _) h1 _ _

theorem bnd_query {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {lv : Flock.Level} {rows : Array (Array F128)}
    {fold : Array F256} {st : LState} {t : Transcript} (h : Bnd B N t) (hq : lv.q + 1 ≤ B) :
    Bnd B N (query p cf lv rows fold st t).2.2 :=
  Bnd.afterSq ((Bnd.afterSq (h.ap _) (by omega) _ _).ap _)
    (by have := Nat.log2_le_self (lv.q - 1); omega) _ _

theorem bnd_close {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {cols start : ℕ}
    {r : Array ℕ × Array F128 × F256} {st : LState} {t : Transcript} (h : Bnd B N t) (h1 : 1 ≤ B) :
    Bnd B N (close p cf cols start r st t).2 :=
  Bnd.afterSq ((h.ap _).ap _) h1 _ _

theorem bnd_closeLast {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {cols start : ℕ}
    {r : Array ℕ × Array F128 × F256} {st : LState} {t : Transcript} (h : Bnd B N t) (h1 : 1 ≤ B) :
    Bnd B N (closeLast p cf cols start r st t).2 :=
  Bnd.afterSq (h.ap _) h1 _ _

theorem bnd_head {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {n : ℕ} {st : LState} {t : Transcript}
    (h : Bnd B N t) (hn : n ≤ B) (h1 : 1 ≤ B) : Bnd B N (head p cf n st t).2 :=
  (Bnd.afterSq (((h.afterSq hn _ _).ap _).ap _) h1 _ _).ap _

theorem bnd_lv0 {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {initCh : Array F256}
    {st : LState} {t : Transcript} (h : Bnd B N t) (h2 : 2 ≤ B) (hn : sch.n + 1 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) : Bnd B N (lv0 p cf sch initCh st t).2 := by
  unfold lv0
  exact bnd_close (bnd_query (bnd_ood (bnd_ood (h.ap _) (by omega) (by omega)) (by omega) (by omega)) (hq 0))
    (by omega)

theorem bnd_stepN {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {i : ℕ}
    {x : Vars × LState × Transcript} (h : Bnd B N x.2.2) (hs : x.1.1 ≤ B) (h2 : 2 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) :
    (stepN p cf sch i x).1.1 ≤ B ∧ Bnd B N (stepN p cf sch i x).2.2 := by
  refine ⟨show x.1.1 - 4 + 1 ≤ B by omega, ?_⟩
  unfold stepN
  exact bnd_close (bnd_query (bnd_ood (bnd_ood ((bnd_chals h h2 4).ap _) (by omega) (by omega)) (by omega)
    (by omega)) (hq _)) (by omega)

theorem bnd_stepL {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule}
    {x : Vars × LState × Transcript} (h : Bnd B N x.2.2) (hs : x.1.1 ≤ B) (h2 : 2 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) :
    (stepL p cf sch x).1.1 ≤ B ∧ Bnd B N (stepL p cf sch x).2.2 := by
  refine ⟨hs, ?_⟩
  unfold stepL
  exact bnd_closeLast (bnd_query ((bnd_chals h h2 4).ap _) (hq _)) (by omega)

theorem bnd_loop {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule}
    {x : Vars × LState × Transcript} (h : Bnd B N x.2.2) (hs : x.1.1 ≤ B) (h2 : 2 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) :
    ∀ j, (loop p cf sch x j).1.1 ≤ B ∧ Bnd B N (loop p cf sch x j).2.2
  | 0 => ⟨hs, h⟩
  | j + 1 => by
    obtain ⟨hs', h'⟩ := bnd_loop h hs h2 hq j
    show (step p cf sch j _).1.1 ≤ B ∧ Bnd B N (step p cf sch j _).2.2
    unfold step
    split
    · exact bnd_stepN h' hs' h2 hq
    · exact bnd_stepL h' hs' h2 hq

/-- **Ligerito's squeezes are few coins**: `2`, `1`, the schedule's `n` and split dimensions, and each level's `q`
and `⌊log₂(q − 1)⌋ + 1`. -/
theorem bnd_out {B N : ℕ} {p : LigeritoProof} {cf : ℕ → ℕ → F128} {sch : Schedule} {T : F128}
    {cap : Array ByteArray} {t : Transcript} (h : Bnd B N t) (h2 : 2 ≤ B) (hn : sch.n + 1 ≤ B)
    (hq : ∀ j, (sch.levels.getD j default).q + 1 ≤ B) : Bnd B N (out p cf sch T cap t).2.2 := by
  have hpre : Bnd B N (pre p cf sch T cap t).2.2 := by
    unfold pre
    exact bnd_lv0 (bnd_chals (bnd_head (((h.ap _).ap _).ap _) (by omega) (by omega)) h2 _) h2 hn hq
  exact (bnd_loop (x := x₀ p cf sch T cap t) hpre (show sch.n - sch.k0 + 1 ≤ B by omega) h2 hq _).2

end Lg

theorem fast100_lt {m : ℕ} {s : Schedule} (h : fast100 m = some s) : m < 40 := by
  by_contra hm
  rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at h
  cases h

/-- fast100's schedules squeeze few coins: `n + 1` and every level's `q + 1` are at most `65536`. -/
theorem fast100_small {m : ℕ} {s : Schedule} (h : fast100 m = some s) :
    s.n + 1 ≤ 65536 ∧ ∀ j, (s.levels.getD j default).q + 1 ≤ 65536 := by
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide (s.n + 1 ≤ 65536 ∧
      ∀ j < s.levels.size, (s.levels.getD j default).q + 1 ≤ 65536) = true := by
    decide +kernel
  obtain ⟨hn, hq⟩ := of_decide_eq_true (key ⟨m, fast100_lt h⟩ s h)
  refine ⟨hn, fun j => ?_⟩
  by_cases hj : j < s.levels.size
  · exact hq j hj
  · rw [Array.getD, dite_eq_right (fun h => absurd h hj)]
    decide

/-- **A returning rep records a round, each of at most `65536` coins** (`RepOk`), from the empty transcript, at a
table of `m ≥ 14` with `fast100`'s schedule and at most `65534` extra claims. -/
theorem repP_repOk (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (hsch : fast100 st.m = some sch) (hx : extras.size + 2 ≤ 65536) (d : ByteArray) :
    Ret ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run
      (Transcript.init #[] d)) fun pt => JsonRT.RepOk pt.2.rounds := by
  rintro l ⟨p, t'⟩ l' hrun
  have h40 := fast100_lt hsch
  obtain ⟨hn, hL, -, hq, -⟩ := fast100_ligerito hsch
  obtain ⟨hn', hq'⟩ := fast100_small hsch
  obtain ⟨full, hfold⟩ := repP_fold resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  obtain ⟨-, -, ht', -⟩ := repP_ligerito resp h _ _ _ tbl cm0 _ salt nonce hm hn hL hq _ rfl l hfold hrun
  have h₀ : Bnd 65536 0 (Transcript.init #[] d) := ⟨Nat.zero_le _, by simp [Transcript.init]⟩
  have hz := Zc.bnd_tEnd (m := st.m) (digest := st.digest) (cap := flat cm0.cap)
    (cs := Zc.coinsAt resp l.size) (tr := tr) h₀ (by omega) (by omega)
  have hz1 := (show Bnd 65536 1 _ from ⟨by rw [Zc.tEnd_size]; omega, hz.2⟩)
  refine (show Bnd 65536 1 t' from ?_) |> fun hb => ⟨hb.1, hb.2⟩
  rw [ht']
  refine Lg.bnd_out (Rs.bnd_tEnd (Lc.bnd_tEnd hz1 (by omega)) (by omega) ?_) (by omega) hn' hq'
  simp only [Array.size_append, abcClaims, List.size_toArray, List.length_cons, List.length_nil]
  omega

end FlockComplete.Replay
