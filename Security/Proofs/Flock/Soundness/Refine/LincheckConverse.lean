import Proofs.Flock.Soundness.Refine.Lincheck
import Proofs.Flock.Soundness.Refine.Frames

/-!
# The lincheck accepts when its checks hold (PROTOCOL.md §10)

`lincheck_of_checks` is `lincheck_refines`' converse on the executable's side: the verifier's `lincheck`
(Piop.lean:126–151) returns `.ok`, with the claim it hands on and the transcript it leaves, when

* each round it squeezes issued one coin and answers the bytes the lincheck framed before it (`LcAnswered`: the label
  and `α`, `β`, each round's two messages and its challenge, `z_partial` and `r'`; the chunks of
  `Discharge.Exec.lcFrames`, which `lincheck_emitted` shows an accepting run consumes);
* its rounds (Piop.lean:134–142), on those rounds' coins, end at `dot comb z_partial` with challenges `ts`: the fold of
  `lStep` from `α·v_A + v_B + β` (`Flock.RecAlgebra.lcFold`), which is the check of Piop.lean:148;
* the statement's circuit fold at `α`, `β` and `ts` is `comb` (Piop.lean:143–145), of the width Piop.lean:146 checks.

`lincheck_plays_of_checks` feeds it to `lincheck_refines`: the model's lincheck plays the proof's messages on those
coins and returns the executable's `ab` claim.
-/

namespace FlockSoundness.Refine

open Flock Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- Round `r` issued one coin and answers the bytes `L`. -/
def OneCoin (r : Round) (L : List UInt8) : Prop := r.n = 1 ∧ r.answers ⟨L.toArray⟩

/-- A round that retained the bytes `L` answers them. -/
theorem answers_of_retains {r : Round} {L : List UInt8} (h : Retains r L) : r.answers ⟨L.toArray⟩ := by
  unfold Retains at h
  unfold Round.answers
  revert h
  cases r.bytes with
  | none => intro h; cases h
  | some B =>
    obtain ⟨d⟩ := B
    intro h
    simp at h
    subst h
    simp

/-- **The lincheck's rounds answer its frames.** From `t`, the rounds the lincheck squeezes (Piop.lean:129–151) each
issued one coin and answer what was framed before them: `t`'s pending bytes, the label and `α`'s squeeze; `β`'s
squeeze; per round `j < kLog − 6` its messages `q1`, `qinf` and its challenge's squeeze; `z_partial` and `r'`'s
squeeze. -/
structure LcAnswered (t : Transcript) (kLog : ℕ) (p : LincheckProof) : Prop where
  alpha : OneCoin t.rounds[t.k]! (t.pending.data.toList ++ encs [.label "flock-lincheck-v0".toUTF8, .sq 1 1])
  beta : OneCoin t.rounds[t.k + 1]! (encs [.sq 1 1])
  rounds : ∀ j < kLog - 6,
    OneCoin t.rounds[t.k + 2 + j]! (encs [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1])
  final : OneCoin t.rounds[t.k + 2 + (kLog - 6)]! (encs [.vec p.zPartial.toList, .sq 1 1])

section Run

variable {α β : Type}

private theorem bind_of {x : TM α} {f : α → TM β} {t t₁ : Transcript} {a : α} (hx : x t = .ok (a, t₁)) :
    (x >>= f) t = f a t₁ := by
  change (StateT.bind x f) t = _
  unfold StateT.bind
  change Except.bind (x t) _ = _
  rw [hx]
  rfl

private theorem label_of (l : String) (t : Transcript) :
    label l t = .ok ((), { t with pending := t.pending ++ header 2 0 l.toUTF8.size ++ padded l.toUTF8 }) := rfl

private theorem absorbF_of (v : F128) (t : Transcript) :
    absorbF v t = .ok ((), { t with pending := F128.pushBytes (t.pending ++ header 3 1 1) v }) := rfl

private theorem absorbSlice_of (vs : Array F128) (t : Transcript) :
    absorbSlice vs t = .ok ((), { t with pending := vs.foldl F128.pushBytes (t.pending ++ header 3 2 vs.size) }) :=
  rfl

private theorem pure_of (a : α) (t : Transcript) : (pure a : TM α) t = .ok (a, t) := rfl

private theorem check_of {b : Bool} {why : String} {t : Transcript} (hb : b = true) : check b why t = .ok ((), t) := by
  subst hb; rfl

/-- A squeeze of one coin from a round that issued one and answers the framed bytes returns its first coin. -/
private theorem sample_of {t : Transcript} {L : List UInt8} (h : OneCoin t.rounds[t.k]! L)
    (hL : t.pending.data.toList ++ (Frame.sq 1 1).enc = L) :
    sample t = .ok ((t.rounds[t.k]!).coins.getD 0 0, ⟨t.rounds, t.k + 1, ByteArray.empty⟩) := by
  obtain ⟨hn, ha⟩ := h
  have hlt : t.k < t.rounds.size := by
    by_contra hge
    rw [getElem!_neg t.rounds t.k hge] at hn
    exact absurd hn (by decide)
  have hget : t.rounds[t.k]? = some t.rounds[t.k]! := by
    rw [getElem!_pos t.rounds t.k hlt]; exact Array.getElem?_eq_getElem hlt
  have hc : framed t 1 1 = ⟨L.toArray⟩ := by
    rw [← hL, ← framed_enc]
  have ha' : (t.rounds[t.k]!).answers (framed t 1 1) := by rw [hc]; exact ha
  have hsq : squeeze 1 1 t = .ok ((t.rounds[t.k]!).coins, ⟨t.rounds, t.k + 1, ByteArray.empty⟩) := by
    simp [squeeze, hget, hn, ha']
  unfold sample
  rw [bind_of hsq]
  rfl

/-- **A loop answering one round per step.** If every step of the body, with nothing pending, consumes the next round,
which meets `Q`, and yields `step i s c` for its coins `c`, leaving nothing pending, then the loop over
`a, …, a + n − 1` from round `base + a` yields the fold of `step` over rounds `base + a, …, base + a + n − 1`. -/
private theorem forIn_of_rounds {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
    (Q : ℕ → Round → Prop)
    (hbody : ∀ i s (t : Transcript), t.pending = ByteArray.empty → Q i t.rounds[t.k]! →
      body i s t = .ok (.yield (step i s (t.rounds[t.k]!).coins), ⟨t.rounds, t.k + 1, ByteArray.empty⟩)) :
    ∀ (n a base : ℕ) (s : σ) (rs : Array Round), (∀ i ∈ List.range' a n, Q i rs[base + i]!) →
      (forIn (List.range' a n) s body : TM σ) ⟨rs, base + a, ByteArray.empty⟩ =
        .ok ((List.range' a n).foldl (fun s i => step i s (rs[base + i]!).coins) s,
          ⟨rs, base + a + n, ByteArray.empty⟩)
  | 0, a, base, s, rs, _ => by
      simp only [List.range'_zero, List.forIn_nil, List.foldl_nil]
      rfl
  | n + 1, a, base, s, rs, hQ => by
      rw [forIn_range'_cons, bind_of (hbody a s ⟨rs, base + a, ByteArray.empty⟩ rfl
        (by exact hQ a (by rw [List.range'_succ]; exact List.mem_cons_self)))]
      have ih := forIn_of_rounds body step Q hbody n (a + 1) base (step a s (rs[base + a]!).coins) rs
        (fun i hi => hQ i (by rw [List.range'_succ]; exact List.mem_cons_of_mem _ hi))
      rw [List.range'_succ, List.foldl_cons, show base + a + (n + 1) = base + (a + 1) + n by omega]
      exact ih

private theorem forIn_range_of {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
    (Q : ℕ → Round → Prop)
    (hbody : ∀ i s (t : Transcript), t.pending = ByteArray.empty → Q i t.rounds[t.k]! →
      body i s t = .ok (.yield (step i s (t.rounds[t.k]!).coins), ⟨t.rounds, t.k + 1, ByteArray.empty⟩))
    (n base : ℕ) (s out : σ) (rs : Array Round) (hQ : ∀ i < n, Q i rs[base + i]!)
    (hout : (List.range' 0 n).foldl (fun s i => step i s (rs[base + i]!).coins) s = out) :
    (forIn [:n] s body : TM σ) ⟨rs, base, ByteArray.empty⟩ = .ok (out, ⟨rs, base + n, ByteArray.empty⟩) := by
  rw [forIn_range_eq, ← hout]
  exact forIn_of_rounds body step Q hbody n 0 base s rs (fun i hi => hQ i (by rw [List.mem_range'_1] at hi; omega))

end Run

/-- **The lincheck accepts when its checks hold** (`lincheck_refines`' converse, Piop.lean:126–151). From a
transcript `t` whose rounds answer the lincheck's frames (`LcAnswered`), with `α`, `β`, the rounds' coins `C j` and `r'`
the first coins of rounds `t.k`, `t.k + 1`, `t.k + 2 + j` and `t.k + 2 + (k_log − 6)`: when the rounds (the fold
of `lStep` on `p`'s round messages and `C`, `Flock.RecAlgebra.lcFold`) from `α·v_A + v_B + β` end at
`dot comb z_partial` with challenges `ts` (Piop.lean:148), and the circuit fold at `α`, `β`, `ts` is `comb` of width
64 (Piop.lean:143–146), the verifier's lincheck returns `w = dot (skipWeights r') z_partial`, `r'` and `ts`
reversed, having consumed its `k_log − 3` rounds with nothing left pending. -/
theorem lincheck_of_checks (c : CircuitFold) (zc : Flock.ZcOut) (p : LincheckProof) (t : Transcript)
    (ha : LcAnswered t c.kLog p) {ts comb : Array F128}
    (hfold : (List.range' 0 (c.kLog - 6)).foldl (fun s j => lStep p.rounds j s (t.rounds[t.k + 2 + j]!).coins)
        ((t.rounds[t.k]!).coins.getD 0 0 * zc.vA + zc.vB + (t.rounds[t.k + 1]!).coins.getD 0 0, #[]) =
      (dot comb p.zPartial, ts))
    (hcomb : c.folded ((t.rounds[t.k]!).coins.getD 0 0) ((t.rounds[t.k + 1]!).coins.getD 0 0) (skipWeights zc.z)
      (zc.rho.extract 0 (c.kLog - 6)) ts = .ok comb)
    (hsize : comb.size = 64) :
    Flock.lincheck c zc p t =
      .ok ({ w := dot (skipWeights ((t.rounds[t.k + 2 + (c.kLog - 6)]!).coins.getD 0 0)) p.zPartial,
             rPrime := (t.rounds[t.k + 2 + (c.kLog - 6)]!).coins.getD 0 0, rIn := ts.reverse },
        ⟨t.rounds, t.k + 2 + (c.kLog - 6) + 1, ByteArray.empty⟩) := by
  obtain ⟨hα, hβ, hR, hr⟩ := ha
  unfold Flock.lincheck
  dsimp only
  refine (bind_of (label_of _ _)).trans ?_
  refine (bind_of (sample_of (by exact hα) (by dsimp only; rw [label_enc]; simp [encs]))).trans ?_
  refine (bind_of (sample_of (by exact hβ) (by simp [encs]))).trans ?_
  refine (bind_of (forIn_range_of _ (lStep p.rounds)
    (fun j r => OneCoin r (encs [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1])) ?hb (c.kLog - 6) _ _
    (dot comb p.zPartial, ts) _ (fun j hj => by exact hR j hj) (by exact hfold))).trans ?_
  case hb =>
    intro j s tt hp hq
    obtain ⟨cl, tss⟩ := s
    rcases hg : p.rounds[j]! with ⟨g1, ginf⟩
    dsimp only
    refine (bind_of (absorbF_of _ _)).trans ?_
    refine (bind_of (absorbF_of _ _)).trans ?_
    refine (bind_of (sample_of (by exact hq)
      (by dsimp only; rw [absorbF_enc, absorbF_enc]; simp [encs, hp, hg]))).trans ?_
    unfold lStep
    rw [hg]
    rfl
  dsimp only
  rw [hcomb]
  dsimp only
  refine (bind_of (pure_of _ _)).trans ?_
  refine (bind_of (check_of (by simp [hsize]))).trans ?_
  refine (bind_of (absorbSlice_of _ _)).trans ?_
  refine (bind_of (check_of (by simp))).trans ?_
  refine (bind_of (sample_of (by exact hr) (by dsimp only; rw [absorbSlice_enc]; simp [encs]))).trans ?_
  rfl

open Game FlockLevel3 in
/-- **The lincheck's plays from its checks.** Under `lincheck_of_checks`' conditions, on a circuit fold realizing `S`
and a zerocheck output that is the model's `zc'`, the model's lincheck, played with the proof's lincheck messages and
the coins of rounds `t.k, …, t.k + 2 + (k_log − 6)`, returns the claim `lincheck_of_checks` hands on, as `ab`
(`lincheck_refines`). -/
theorem lincheck_plays_of_checks (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S)
    (zc : Flock.ZcOut) (zc' : Model.ZcOut 𝔽 S.m) (hzc : ZcRel S.m zc zc') (p : LincheckProof) (t : Transcript)
    (hwf : RoundsWF t.rounds) (ha : LcAnswered t c.kLog p) {ts comb : Array F128}
    (hfold : (List.range' 0 (c.kLog - 6)).foldl (fun s j => lStep p.rounds j s (t.rounds[t.k + 2 + j]!).coins)
        ((t.rounds[t.k]!).coins.getD 0 0 * zc.vA + zc.vB + (t.rounds[t.k + 1]!).coins.getD 0 0, #[]) =
      (dot comb p.zPartial, ts))
    (hcomb : c.folded ((t.rounds[t.k]!).coins.getD 0 0) ((t.rounds[t.k + 1]!).coins.getD 0 0) (skipWeights zc.z)
      (zc.rho.extract 0 (c.kLog - 6)) ts = .ok comb)
    (hsize : comb.size = 64) :
    ∃ ab, Plays fOf (Model.lincheck execArith S zc') (lcMsgs S.kLog p)
        (consumed t.rounds t.k (t.k + 2 + (c.kLog - 6) + 1)) (some ab) ∧
      ClaimRel S.m (abcClaims S.m S.kLog zc
        { w := dot (skipWeights ((t.rounds[t.k + 2 + (c.kLog - 6)]!).coins.getD 0 0)) p.zPartial,
          rPrime := (t.rounds[t.k + 2 + (c.kLog - 6)]!).coins.getD 0 0, rIn := ts.reverse })[0]! ab :=
  (lincheck_refines S c hc zc zc' hzc p t _ _ hwf (lincheck_of_checks c zc p t ha hfold hcomb hsize)).2.2

end FlockSoundness.Refine
