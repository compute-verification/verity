import Proofs.Flock.Soundness.Refine.Rep
import Proofs.Flock.Soundness.Refine.LincheckConverse
import Proofs.Flock.Soundness.Refine.OpeningConverse
import Proofs.Flock.VBridge.Algebra.ZerocheckConverse

/-!
# One rep accepts when its stages' checks hold (PROTOCOL.md §9–§14)

`verifyRep_of_checks` is `rep_refines`' converse on the executable's side: the verifier's per-rep check `verifyRep`
(Verify.lean:37–55) returns the proof's cap when the record carries the rep's stream, the statement's extra claims and
the proof decode, the PCS parameters are the pinned ones, the rep's transcript program (`repProg`, Verify.lean:46–50)
returns on the stream's replay transcript, every recorded round was consumed, and the cap is the committed `root_B`.

`prog_of_checks` gives the program's run from the stages' converses, each from the transcript the stage before left:
`RecAlgebra.bindAndZerocheck_of_checks`, `lincheck_of_checks` (with the zerocheck's final values pending) and
`ringSwitch_run`. Ligerito's run enters as one hypothesis, at the batched value, basis and transcript ring switching
returns (`rsT`, `rsSws`).

`roundsOf` builds a record's rounds from their coin counts, coins and retained bytes; each round retains its bytes
(`roundsOf_retains`), so the stages' frame hypotheses hold by construction at the bytes the stages frame.
-/

namespace FlockSoundness.Refine

open Flock Transcript

/-- The rep's transcript program (Verify.lean:46–50). -/
def repProg (st : Setup) (sch : Flock.Schedule) (p : Proof) (extras : Array Claim) : TM Unit := do
  let zc ← Flock.bindAndZerocheck st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
  let lc ← Flock.lincheck st.fold zc p.lincheck
  let (T, sws) ← Flock.ringSwitch st.m (Flock.abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
  Flock.ligerito st.merkle sch T p.cap p.ligerito (Flock.basis sws)

/-- The zerocheck's output at points `r`, `z`, `ρ` (Piop.lean:102). -/
abbrev zcOf (p : ZerocheckProof) (r : Array F128) (z : F128) (rho : Array F128) : Flock.ZcOut :=
  { z := z, rho := rho, rRest := r, vA := p.finalA, vB := p.finalB, vC := p.finalC }

/-- The lincheck's output at `r'` and challenges `ts` (Piop.lean:151). -/
abbrev lcOf (p : LincheckProof) (rp : F128) (ts : Array F128) : Flock.LcOut :=
  { w := dot (skipWeights rp) p.zPartial, rPrime := rp, rIn := ts.reverse }

/-- Ring switching's batched value `T = Σ_k γ_k·β_k` (Opening.lean:88). -/
abbrev rsT (claims : Array Claim) (rs : Array (Array F128)) (rpp : ℕ → Array F128) (gamma : Array F128) : F128 :=
  dot gamma ((List.range claims.size).map fun k => dot (eqTable (rpp k)) (tOf rs[k]!)).toArray

/-- Ring switching's switched claims (Opening.lean:89). -/
abbrev rsSws (claims : Array Claim) (rpp : ℕ → Array F128) (gamma : Array F128) : Array Switched :=
  (Array.range claims.size).map fun k => ({ gamma := gamma[k]!, rpp := rpp k, z := claims[k]!.x.extract 1 } : Switched)

/-- **The rep's program returns when its stages' checks hold** (Verify.lean:46–50). From `t`:
* the zerocheck's rounds, messages and checks, as `RecAlgebra.bindAndZerocheck_of_checks` takes them, on the digest,
  the flattened cap and `p.zerocheck`, at points `r`, `z`, `ρ`;
* from round `k₁ = t.k + 3 + (m − 6)`, with the zerocheck's `final_a`, `final_b` pending, the lincheck's rounds and
  checks, as `lincheck_of_checks` takes them, at the zerocheck's output;
* from round `k₂ = k₁ + 2 + (k_log − 6) + 1`, with nothing pending, ring switching's rounds and checks, as
  `ringSwitch_run` takes them, on the claims `abcClaims m k_log zc lc ++ extras`;
* Ligerito's run from round `k₂ + n + 1` (`n` the claims), at ring switching's `T` and switched claims.
Then the program returns, at the transcript Ligerito's run leaves. -/
theorem prog_of_checks (st : Setup) (sch : Flock.Schedule) (p : Proof) (extras : Array Claim) (t : Transcript)
    (hm : 14 ≤ st.m)
    -- the zerocheck, from `t`
    (hn₀ : (t.rounds[t.k]!).n = 6) (hn₁ : (t.rounds[t.k + 1]!).n = st.m - 13) (hn₂ : (t.rounds[t.k + 2]!).n = 1)
    (hnr : ∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1)
    (ha₀ : RecAlgebra.AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
      .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6]))
    (ha₁ : RecAlgebra.AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)]))
    (ha₂ : RecAlgebra.AnswersL (t.rounds[t.k + 2]!)
      (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1]))
    (har : ∀ i < st.m - 6, RecAlgebra.AnswersL (t.rounds[t.k + 3 + i]!)
      (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1]))
    {r : Array F128} {z : F128} {rho : Array F128} {C : ℕ → Array F128}
    (hr : r = pinned ++ (t.rounds[t.k + 1]!).coins) (hz : z = (t.rounds[t.k + 2]!).coins.getD 0 0)
    (hC : ∀ i < st.m - 6, (C i).getD 0 0 = (t.rounds[t.k + 3 + i]!).coins.getD 0 0)
    (hpc : dot (lagrange nodesL den6Inv z) p.zerocheck.round1c = p.zerocheck.finalC)
    (hfold : RecAlgebra.zcFold r p.zerocheck C (st.m - 6) (RecAlgebra.zcStart p.zerocheck z) =
      (p.zerocheck.finalA * p.zerocheck.finalB, rho))
    -- the lincheck, from round `k₁`
    {k₁ : ℕ} (hk₁ : k₁ = t.k + 3 + (st.m - 6))
    (hlc : LcAnswered ⟨t.rounds, k₁, ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩
      st.fold.kLog p.lincheck)
    {ts comb : Array F128}
    (hlfold : (List.range' 0 (st.fold.kLog - 6)).foldl
        (fun s j => lStep p.lincheck.rounds j s (t.rounds[k₁ + 2 + j]!).coins)
        ((t.rounds[k₁]!).coins.getD 0 0 * p.zerocheck.finalA + p.zerocheck.finalB +
          (t.rounds[k₁ + 1]!).coins.getD 0 0, #[]) =
      (dot comb p.lincheck.zPartial, ts))
    (hcomb : st.fold.folded ((t.rounds[k₁]!).coins.getD 0 0) ((t.rounds[k₁ + 1]!).coins.getD 0 0) (skipWeights z)
      (rho.extract 0 (st.fold.kLog - 6)) ts = .ok comb)
    (hsize : comb.size = 64)
    -- ring switching, from round `k₂`
    {k₂ : ℕ} (hk₂ : k₂ = k₁ + 2 + (st.fold.kLog - 6) + 1)
    {claims : Array Claim}
    (hcl : claims = Flock.abcClaims st.m st.kLog (zcOf p.zerocheck r z rho)
      (lcOf p.lincheck ((t.rounds[k₁ + 2 + (st.fold.kLog - 6)]!).coins.getD 0 0) ts) ++ extras)
    {rpp : ℕ → Array F128} {gamma : Array F128} (hrs : p.ringSwitches.size = claims.size)
    (hshape : ∀ k < claims.size,
      p.ringSwitches[k]!.size = 128 ∧ claims[k]!.skip.size = 64 ∧ claims[k]!.x.size = st.m - 6)
    (hval : ∀ k < claims.size, dot (wOf claims[k]!) p.ringSwitches[k]! = claims[k]!.value)
    (hR : RsRounds ⟨t.rounds, k₂, ByteArray.empty⟩ p.ringSwitches claims.size rpp gamma)
    -- Ligerito, from round `k₂ + n + 1`
    {t₄ : Transcript}
    (hlg : Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
      (Flock.basis (rsSws claims rpp gamma)) ⟨t.rounds, k₂ + claims.size + 1, ByteArray.empty⟩ = .ok ((), t₄)) :
    repProg st sch p extras t = .ok ((), t₄) := by
  subst hk₁ hk₂ hcl
  obtain ⟨t₁, h₁, hr₁, hk₁, hp₁⟩ := RecAlgebra.bindAndZerocheck_of_checks st.m hm st.digest
    (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck t hn₀ hn₁ hn₂ hnr ha₀ ha₁ ha₂ har hr hz hC hpc hfold
  have e : t₁ = ⟨t.rounds, t.k + 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ := by
    obtain ⟨R, K, ⟨d⟩⟩ := t₁
    dsimp only at hr₁ hk₁ hp₁
    subst hr₁ hk₁
    rw [← hp₁]
  subst e
  have hn : (Flock.abcClaims st.m st.kLog (zcOf p.zerocheck r z rho)
      (lcOf p.lincheck ((t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6)]!).coins.getD 0 0) ts) ++
        extras).size ≠ 0 := by
    simp [Flock.abcClaims]
  unfold repProg
  refine Eq.trans (bind_run h₁) ?_
  refine Eq.trans (bind_run (lincheck_of_checks st.fold _ p.lincheck _ hlc hlfold hcomb hsize)) ?_
  refine Eq.trans (bind_run (ringSwitch_run st.m _ p.ringSwitches _ rpp gamma hn hrs hshape hval hR)) ?_
  exact hlg

/-- **One rep accepts when its stages' checks hold** (Verify.lean:37–55). When the record carries table `i`'s rep `r`
stream, the statement's extra claims at the link points are `extras`, the proof decodes to `p` at the shape
`verifyRep` decodes with, its PCS parameters are the pinned ones, the rep's program returns from the stream's replay
transcript at `t₄` (`prog_of_checks`), `t₄` consumed every recorded round, and the proof's cap is the committed
`root_B`, then `verifyRep` returns that cap. -/
theorem verifyRep_of_checks (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (sess : Session) (i r : ℕ)
    (bytes : ByteArray) (stream : StreamRec) (extras : Array Claim) (p : Proof) (t₄ : Transcript)
    (hs : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) =
      some stream)
    (hex : st.extra sess.points = .ok extras)
    (hdec : decodeProof (shapeOf st sch extras.size) bytes = .ok p)
    (hpar : (p.params != pcsParams st.merkle sch) = false)
    (hprog : repProg st sch p extras (Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8) =
      .ok ((), t₄))
    (hk : t₄.k = stream.rounds.size)
    (hcap : (p.cap.foldl (· ++ ·) ByteArray.empty == sess.rootB[i]!) = true) :
    verifyRep spec st sch sess i r bytes = .ok (p.cap.foldl (· ++ ·) ByteArray.empty) := by
  have ebind : ∀ {α β : Type} {x : Except String α} {f : α → Except String β} {a : α}, x = .ok a →
      (x >>= f) = f a := by
    intro α β x f a h; rw [h]; rfl
  unfold verifyRep
  dsimp only
  rw [hs]
  refine Eq.trans (ebind hex) ?_
  refine Eq.trans (ebind hdec) ?_
  rw [ite_eq_right (by rw [hpar]; exact Bool.false_ne_true)]
  refine Eq.trans (ebind hprog) ?_
  change (if (t₄.k != stream.rounds.size) = true then _ else _) = _
  rw [ite_eq_right (by simp [hk])]
  rw [ite_eq_right (by simp [bne, hcap])]
  rfl

/-! ## A record's rounds from their coins and bytes -/

/-- The rounds whose coin counts, coins and retained bytes are `l`'s, in order. -/
def roundsOf (l : List (ℕ × Array F128 × List UInt8)) : Array Round :=
  (l.map fun x => ({ n := x.1, sha := ByteArray.empty, coins := x.2.1, bytes := some ⟨x.2.2.toArray⟩ } : Round)).toArray

theorem roundsOf_size (l : List (ℕ × Array F128 × List UInt8)) : (roundsOf l).size = l.length := by
  simp [roundsOf]

theorem roundsOf_get (l : List (ℕ × Array F128 × List UInt8)) {i : ℕ} (h : i < l.length) :
    (roundsOf l)[i]! = { n := l[i].1, sha := ByteArray.empty, coins := l[i].2.1, bytes := some ⟨l[i].2.2.toArray⟩ } := by
  rw [getElem!_pos (roundsOf l) i (by rw [roundsOf_size]; exact h)]
  simp [roundsOf]

theorem roundsOf_n (l : List (ℕ × Array F128 × List UInt8)) {i : ℕ} (h : i < l.length) :
    ((roundsOf l)[i]!).n = l[i].1 := by
  rw [roundsOf_get l h]

theorem roundsOf_coins (l : List (ℕ × Array F128 × List UInt8)) {i : ℕ} (h : i < l.length) :
    ((roundsOf l)[i]!).coins = l[i].2.1 := by
  rw [roundsOf_get l h]

/-- Each round retains its bytes. -/
theorem roundsOf_retains (l : List (ℕ × Array F128 × List UInt8)) {i : ℕ} (h : i < l.length) :
    Retains ((roundsOf l)[i]!) l[i].2.2 := by
  rw [roundsOf_get l h]
  simp [Retains]

/-- Each round answers the framed bytes whose bytes it retained. -/
theorem roundsOf_answers (l : List (ℕ × Array F128 × List UInt8)) {i : ℕ} (h : i < l.length) {c : ByteArray}
    (hc : c.data.toList = l[i].2.2) : ((roundsOf l)[i]!).answers c :=
  RecAlgebra.answersL_of_retains (roundsOf_retains l h) c hc

theorem roundsOf_oneCoin (l : List (ℕ × Array F128 × List UInt8)) {i : ℕ} (h : i < l.length) (hn : l[i].1 = 1) :
    OneCoin ((roundsOf l)[i]!) l[i].2.2 :=
  ⟨by rw [roundsOf_n l h, hn], answers_of_retains (roundsOf_retains l h)⟩

/-- Rounds whose coin arrays have their coin counts are well formed. -/
theorem roundsOf_wf (l : List (ℕ × Array F128 × List UInt8)) (h : ∀ x ∈ l, x.2.1.size = x.1) :
    RoundsWF (roundsOf l) := by
  intro i hi
  rw [roundsOf_size] at hi
  rw [roundsOf_coins l hi, roundsOf_n l hi]
  exact h _ (List.getElem_mem hi)

end FlockSoundness.Refine
