import Proofs.Flock.Soundness.Refine.Opening

/-!
# Ring switching's checks make the executable's ring-switch stage return (PROTOCOL.md §12)

The converse of `opening_refines`' first step. `ringSwitch_run`: when every claim's ring switch passes the verifier's
value check (`dot w s == c.value`, Opening.lean:78), and the record holds ring switching's rounds from
`t` (`RsRounds`: claim `k`'s round issued 7 coins `rpp k`, the batching round issued `n` coins `gamma`, and each answers
the bytes framed before it), then `ringSwitch` (Opening.lean:66–88) run from `t` returns the batched value
`T = Σ_k γ_k·β_k` with `β_k = Σ_u eq(rpp k, u)·t_k[u]` (`Refine.tOf`), the switched claims, and `t` with the `n + 1`
rounds consumed. `opening_of_checks` then applies `opening_refines` to that run: the model's opening, played with the
ring switches and the coins `rpp 0, …, rpp (n − 1), gamma`, returns the executable's batched value and basis.

The run is taken one step forward at a time (`bind_run`): a label or an absorb appends its frame to the pending bytes,
a passing check changes nothing, a squeeze of `n ≠ 0` coins consumes the next round when that round issued `n` coins and
answers the framed bytes (Transcript.lean:75–90), and a loop whose every step yields returns the fold of its steps
(`forIn_range_run`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript

local notation "𝔽" => FlockLevel3.GF128

variable {α β : Type}

/-! ## One step forward -/

theorem bind_run {x : TM α} {f : α → TM β} {t t₁ : Transcript} {a : α} (h : x t = .ok (a, t₁)) :
    (x >>= f) t = f a t₁ := by
  change (StateT.bind x f) t = _
  unfold StateT.bind
  change Except.bind (x t) _ = _
  rw [h]
  rfl

theorem label_run (l : String) (t : Transcript) :
    label l t = .ok ((), { t with pending := t.pending ++ header 2 0 l.toUTF8.size ++ padded l.toUTF8 }) := rfl

theorem absorbSlice_run (vs : Array F128) (t : Transcript) :
    absorbSlice vs t = .ok ((), { t with pending := vs.foldl F128.pushBytes (t.pending ++ header 3 2 vs.size) }) :=
  rfl

theorem check_run {c : Bool} {why : String} {t : Transcript} (hc : c = true) : check c why t = .ok ((), t) := by
  subst hc; rfl

/-- A squeeze of `n ≠ 0` coins (Transcript.lean:75–90) returns round `t.k`'s coins when that round issued `n` coins and
answers the framed bytes. -/
theorem sampleVec_run {n : ℕ} (hn : n ≠ 0) {t : Transcript} {c : Array F128} (hlt : t.k < t.rounds.size)
    (hN : (t.rounds[t.k]!).n = n) (hc : (t.rounds[t.k]!).coins = c) (ha : (t.rounds[t.k]!).answers (framed t 2 n)) :
    sampleVec n t = .ok (c, { t with pending := ByteArray.empty, k := t.k + 1 }) := by
  unfold sampleVec squeeze
  rw [getElem!_pos t.rounds t.k hlt] at hN hc ha
  simp [hn, Array.getElem?_eq_getElem hlt, hN, ha, hc]

/-- **A loop whose every step yields.** If step `i` of the body, run from `T i`, yields `step i s (C i)` and leaves
`T (i + 1)`, the loop over `a, …, a + n − 1` run from `T a` returns the fold of `step` and leaves `T (a + n)`. -/
theorem forIn_range'_run {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
    (C : ℕ → Array F128) (T : ℕ → Transcript) :
    ∀ (n a : ℕ) (s : σ), (∀ i s, a ≤ i → i < a + n → body i s (T i) = .ok (.yield (step i s (C i)), T (i + 1))) →
      forIn (List.range' a n) s body (T a) =
        .ok ((List.range' a n).foldl (fun s i => step i s (C i)) s, T (a + n))
  | 0, a, s, _ => by
      simp only [List.range'_zero, List.forIn_nil, List.foldl_nil, Nat.add_zero]
      rfl
  | n + 1, a, s, h => by
      rw [forIn_range'_cons]
      refine Eq.trans (bind_run (h a s (Nat.le_refl a) (by omega))) ?_
      rw [List.range'_succ, List.foldl_cons, show a + (n + 1) = a + 1 + n by omega]
      exact forIn_range'_run body step C T n (a + 1) _ fun i s h1 h2 => h i s (by omega) (by omega)

theorem forIn_range_run {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
    (C : ℕ → Array F128) (T : ℕ → Transcript) (n : ℕ) (s : σ) {t : Transcript} (ht : t = T 0)
    (h : ∀ i s, i < n → body i s (T i) = .ok (.yield (step i s (C i)), T (i + 1))) :
    (forIn [:n] s body : TM σ) t = .ok ((List.range' 0 n).foldl (fun s i => step i s (C i)) s, T n) := by
  subst ht
  rw [forIn_range_eq]
  have e := forIn_range'_run body step C T n 0 s fun i s _ hi => h i s (by omega)
  rwa [Nat.zero_add] at e

/-! ## Ring switching's rounds -/

/-- The pending bytes after `label l` (Transcript.lean:42). -/
def labelled (p : ByteArray) (l : String) : ByteArray := p ++ header 2 0 l.toUTF8.size ++ padded l.toUTF8

/-- What is pending when claim `k`'s step begins (Opening.lean:68, 79): the batching label after `p` before the first
claim, nothing after a squeeze. -/
def rsPending (p : ByteArray) : ℕ → ByteArray
  | 0 => labelled p "flock-pcs-open-batch-v0"
  | _ + 1 => ByteArray.empty

/-- The bytes claim `k`'s coins `r''` answer (Opening.lean:74–79): what was pending, the ring-switch label, the ring
switch absorbed as one slice, and the squeeze's header. -/
def rsFramed (p : ByteArray) (s : Array F128) : ByteArray :=
  s.foldl F128.pushBytes (labelled p "flock-ring-switch-v0" ++ header 3 2 s.size) ++ header 4 2 7

/-- The transcript when claim `k`'s step begins: `k` rounds consumed from `t`. -/
def rsAt (t : Transcript) (k : ℕ) : Transcript := { t with pending := rsPending t.pending k, k := t.k + k }

/-- **The record holds ring switching's rounds from `t`**, for `n` claims with ring switches `rs`: round `t.k + k`
issued claim `k`'s 7 coins `rpp k` and answers the bytes framed before them, and round `t.k + n` issued the `n` batching
coins `gamma` and answers the bytes framed before them (Opening.lean:79, 85). -/
structure RsRounds (t : Transcript) (rs : Array (Array F128)) (n : ℕ) (rpp : ℕ → Array F128) (gamma : Array F128) :
    Prop where
  size : t.k + n < t.rounds.size
  rpp_n : ∀ k < n, (t.rounds[t.k + k]!).n = 7
  rpp_coins : ∀ k < n, (t.rounds[t.k + k]!).coins = rpp k
  rpp_answers : ∀ k < n, (t.rounds[t.k + k]!).answers (rsFramed (rsPending t.pending k) rs[k]!)
  gamma_n : (t.rounds[t.k + n]!).n = n
  gamma_coins : (t.rounds[t.k + n]!).coins = gamma
  gamma_answers : (t.rounds[t.k + n]!).answers (ByteArray.empty ++ header 4 2 n)

/-- The rounds ring switching consumes have the coins `rpp 0, …, rpp (n − 1)`, then `gamma`. -/
theorem RsRounds.consumed_eq {t : Transcript} {rs : Array (Array F128)} {n : ℕ} {rpp : ℕ → Array F128}
    {gamma : Array F128} (hR : RsRounds t rs n rpp gamma) :
    consumed t.rounds t.k (t.k + n + 1) = (List.range n).map rpp ++ [gamma] := by
  rw [Nat.add_assoc, consumed_eq_map, List.range'_concat, List.map_append, List.range_eq_range']
  simp only [List.map_cons, List.map_nil, Nat.zero_add, Nat.one_mul, hR.gamma_coins]
  congr 1
  exact List.map_congr_left fun k hk => hR.rpp_coins k (by rw [List.mem_range'_1] at hk; omega)

/-! ## The stage -/

/-- **Ring switching's checks make `ringSwitch` return** (Opening.lean:66–88). When every claim's ring switch passes its
value check `Σ_v w[v]·ŝ[v] = value` (Opening.lean:78), and the record holds ring switching's rounds from `t`
(`RsRounds`), `ringSwitch` returns `T = Σ_k γ_k·β_k` with `β_k = Σ_u eq(rpp k, u)·t_k[u]`, the switched claims
`(γ_k, rpp k, x_k[1 …])`, and `t` with `n + 1` rounds consumed. The run reads neither `hrs` nor `hshape`, the shapes
the proof's decoders and the claims' construction fix. -/
theorem ringSwitch_run (m : ℕ) (claims : Array Flock.Claim) (rs : Array (Array F128)) (t : Transcript)
    (rpp : ℕ → Array F128) (gamma : Array F128) (hn : claims.size ≠ 0) (hrs : rs.size = claims.size)
    (hshape : ∀ k < claims.size, rs[k]!.size = 128 ∧ claims[k]!.skip.size = 64 ∧ claims[k]!.x.size = m - 6)
    (hval : ∀ k < claims.size, dot (wOf claims[k]!) rs[k]! = claims[k]!.value)
    (hR : RsRounds t rs claims.size rpp gamma) :
    ringSwitch m claims rs t =
      .ok ((dot gamma ((List.range claims.size).map fun k => dot (eqTable (rpp k)) (tOf rs[k]!)).toArray,
          (Array.range claims.size).map fun k =>
            ({ gamma := gamma[k]!, rpp := rpp k, z := claims[k]!.x.extract 1 } : Switched)),
        { t with pending := ByteArray.empty, k := t.k + claims.size + 1 }) := by
  have hsz := hR.size
  obtain ⟨hs1, hs2⟩ := oStep_fold claims rs rpp (List.range' 0 claims.size) (#[], #[])
  unfold ringSwitch
  refine Eq.trans (bind_run (label_run _ _)) ?_
  refine Eq.trans (bind_run (forIn_range_run _ (oStep claims rs) rpp (rsAt t) claims.size _ rfl ?_)) ?_
  · intro i st hi
    refine Eq.trans (bind_run (label_run _ _)) ?_
    refine Eq.trans (bind_run (absorbSlice_run _ _)) ?_
    refine Eq.trans (bind_run (check_run ?_)) ?_
    · exact beq_iff_eq.mpr (hval i hi)
    refine Eq.trans (bind_run (sampleVec_run (by decide) (by exact (show t.k + i < t.rounds.size by omega))
      (by exact hR.rpp_n i hi) (by exact hR.rpp_coins i hi) (by exact hR.rpp_answers i hi))) ?_
    rfl
  refine Eq.trans (bind_run (sampleVec_run hn (by exact (show t.k + claims.size < t.rounds.size by omega))
    (by exact hR.gamma_n) (by exact hR.gamma_coins) ?_)) ?_
  · obtain ⟨j, hj⟩ := Nat.exists_eq_succ_of_ne_zero hn
    have e : rsPending t.pending claims.size = ByteArray.empty := by rw [hj]; rfl
    show (t.rounds[t.k + claims.size]!).answers (rsPending t.pending claims.size ++ header 4 2 claims.size)
    rw [e]
    exact hR.gamma_answers
  show Except.ok _ = _
  rw [hs1, hs2]
  have e2 : ∀ k < claims.size, (((#[], #[]) : Array F128 × Array (Array F128 × Array F128)).2 ++
      ((List.range' 0 claims.size).map fun k => (rpp k, claims[k]!.x.extract 1)).toArray)[k]! =
        (rpp k, claims[k]!.x.extract 1) := fun k hk => by
    rw [getElem!_pos _ k (by simp [hk])]
    simp
  rw [List.range_eq_range']
  refine congrArg Except.ok (Prod.ext (Prod.ext (by simp) ?_) rfl)
  apply Array.ext
  · simp
  · intro k h1 _
    have hk : k < claims.size := by simpa using h1
    simp only [Array.getElem_map, Array.getElem_range, e2 k hk]

/-- **Ring switching's checks give the model's opening.** Under `ringSwitch_run`'s hypotheses, on claims that are the
model's `cs'` and a well-formed record, `opening_refines` applies to the run: the model's opening of `cs'`, played with
the ring switches and the coins `rpp 0, …, rpp (n − 1), gamma`, returns the batched value and basis `ringSwitch`
returns. -/
theorem opening_of_checks (m : ℕ) (hm : 7 ≤ m) (claims : Array Flock.Claim) (cs' : List (Model.Claim 𝔽 m))
    (hlen : claims.size = cs'.length) (hcs : ∀ (k : ℕ) (hk : k < cs'.length), ClaimRel m claims[k]! cs'[k])
    (hn : 0 < claims.size) (hn1 : claims.size ≠ 1) (rs : Array (Array F128)) (t : Transcript)
    (hwf : RoundsWF t.rounds) (rpp : ℕ → Array F128) (gamma : Array F128) (hrs : rs.size = claims.size)
    (hshape : ∀ k < claims.size, rs[k]!.size = 128 ∧ claims[k]!.skip.size = 64 ∧ claims[k]!.x.size = m - 6)
    (hval : ∀ k < claims.size, dot (wOf claims[k]!) rs[k]! = claims[k]!.value)
    (hR : RsRounds t rs claims.size rpp gamma) :
    ∃ op, Plays fOf (Model.opening execArith hm cs') (opMsgs claims.size rs)
        ((List.range claims.size).map rpp ++ [gamma]) (some op) ∧
      OpenRel m claims.size
        (dot gamma ((List.range claims.size).map fun k => dot (eqTable (rpp k)) (tOf rs[k]!)).toArray)
        ((Array.range claims.size).map fun k =>
          ({ gamma := gamma[k]!, rpp := rpp k, z := claims[k]!.x.extract 1 } : Switched)) op := by
  obtain ⟨-, -, h⟩ := opening_refines m hm claims cs' hlen hcs hn hn1 rs t _ _ _ hwf
    (ringSwitch_run m claims rs t rpp gamma (by omega) hrs hshape hval hR)
  rw [← hR.consumed_eq]
  exact h

end FlockSoundness.Refine
