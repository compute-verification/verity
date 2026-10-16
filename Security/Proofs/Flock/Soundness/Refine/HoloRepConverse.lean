import Proofs.Flock.Soundness.Refine.HoloRefines
import Proofs.Flock.Soundness.Refine.RepConverse

/-!
# One rep with hidden units accepts when its stages' checks hold

`holoLincheck_of_checks` is `holoLincheck_refines`' converse on the executable's side, `lincheck_of_checks` with the
lincheck's end as a holo record has it: `z_partial`'s round issued no coins and answers `z_partial` and a squeeze of
none, and `r'`'s round issued one and answers `h` and `r'`'s squeeze (`HoloLcAnswered`). The rounds end at
`dot (comb + h) z_partial` (`RecAlgebra.addH`). `holoProg_of_checks` gives `holoRepProg`'s run from the stages'
converses, as `prog_of_checks` gives `repProg`'s; `holoLincheck_plays_of_checks` feeds the lincheck's run to
`holoLincheck_refines`.
-/

namespace FlockSoundness.Refine

open Flock Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- **The lincheck's rounds with hidden units answer its frames**: `LcAnswered`'s rounds up to `z_partial`, whose
round issued no coins and answers `z_partial` and the squeeze of none, then `r'`'s round, which issued one coin and
answers `h` and `r'`'s squeeze. -/
structure HoloLcAnswered (t : Transcript) (kLog : ℕ) (p : LincheckProof) (hpart : Array F128) : Prop where
  alpha : OneCoin t.rounds[t.k]! (t.pending.data.toList ++ encs [.label "flock-lincheck-v0".toUTF8, .sq 1 1])
  beta : OneCoin t.rounds[t.k + 1]! (encs [.sq 1 1])
  rounds : ∀ j < kLog - 6,
    OneCoin t.rounds[t.k + 2 + j]! (encs [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1])
  zp_lt : t.k + 2 + (kLog - 6) < t.rounds.size
  zp_n : (t.rounds[t.k + 2 + (kLog - 6)]!).n = 0
  zp : (t.rounds[t.k + 2 + (kLog - 6)]!).answers ⟨(encs [.vec p.zPartial.toList, .sq 2 0]).toArray⟩
  final : OneCoin t.rounds[t.k + 2 + (kLog - 6) + 1]! (encs [.vec hpart.toList, .sq 1 1])

section Run

private theorem sampleOne_of {t : Transcript} {L : List UInt8} (h : OneCoin t.rounds[t.k]! L)
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
  rw [RecAlgebra.bind_of_ok hsq]
  rfl

private theorem pureOne_of {α : Type} (a : α) (t : Transcript) : (pure a : TM α) t = .ok (a, t) := rfl

private theorem lcForIn_of_rounds {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
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
      rw [forIn_range'_cons, RecAlgebra.bind_of_ok (hbody a s ⟨rs, base + a, ByteArray.empty⟩ rfl
        (by exact hQ a (by rw [List.range'_succ]; exact List.mem_cons_self)))]
      have ih := lcForIn_of_rounds body step Q hbody n (a + 1) base (step a s (rs[base + a]!).coins) rs
        (fun i hi => hQ i (by rw [List.range'_succ]; exact List.mem_cons_of_mem _ hi))
      rw [List.range'_succ, List.foldl_cons, show base + a + (n + 1) = base + (a + 1) + n by omega]
      exact ih

private theorem lcForIn_range_of {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
    (Q : ℕ → Round → Prop)
    (hbody : ∀ i s (t : Transcript), t.pending = ByteArray.empty → Q i t.rounds[t.k]! →
      body i s t = .ok (.yield (step i s (t.rounds[t.k]!).coins), ⟨t.rounds, t.k + 1, ByteArray.empty⟩))
    (n base : ℕ) (s out : σ) (rs : Array Round) (hQ : ∀ i < n, Q i rs[base + i]!)
    (hout : (List.range' 0 n).foldl (fun s i => step i s (rs[base + i]!).coins) s = out) :
    (forIn [:n] s body : TM σ) ⟨rs, base, ByteArray.empty⟩ = .ok (out, ⟨rs, base + n, ByteArray.empty⟩) := by
  rw [forIn_range_eq, ← hout]
  exact lcForIn_of_rounds body step Q hbody n 0 base s rs (fun i hi => hQ i (by rw [List.mem_range'_1] at hi; omega))

end Run

/-- **The lincheck with hidden units accepts when its checks hold** (`holoLincheck_refines`' converse). From a
transcript `t` whose rounds answer the lincheck's frames with hidden units (`HoloLcAnswered`): when the rounds from
`α·v_A + v_B + β` end at `dot (comb + h) z_partial` with challenges `ts`, and the circuit fold at `α`, `β`, `ts` is
`comb` of width 64, `holoLincheckT` returns `w = dot (skipWeights r') z_partial`, `r'` (the first coin of the round
after `z_partial`'s) and `ts` reversed, having consumed its `k_log − 2` rounds with nothing left pending. -/
theorem holoLincheck_of_checks (c : CircuitFold) (zc : Flock.ZcOut) (p : LincheckProof) (hpart : Array F128)
    (t : Transcript) (ha : HoloLcAnswered t c.kLog p hpart) {ts comb : Array F128}
    (hfold : (List.range' 0 (c.kLog - 6)).foldl (fun s j => lStep p.rounds j s (t.rounds[t.k + 2 + j]!).coins)
        ((t.rounds[t.k]!).coins.getD 0 0 * zc.vA + zc.vB + (t.rounds[t.k + 1]!).coins.getD 0 0, #[]) =
      (dot (RecAlgebra.addH comb hpart) p.zPartial, ts))
    (hcomb : c.folded ((t.rounds[t.k]!).coins.getD 0 0) ((t.rounds[t.k + 1]!).coins.getD 0 0) (skipWeights zc.z)
      (zc.rho.extract 0 (c.kLog - 6)) ts = .ok comb)
    (hsize : comb.size = 64) :
    holoLincheckT c zc p hpart t =
      .ok ({ w := dot (skipWeights ((t.rounds[t.k + 2 + (c.kLog - 6) + 1]!).coins.getD 0 0)) p.zPartial,
             rPrime := (t.rounds[t.k + 2 + (c.kLog - 6) + 1]!).coins.getD 0 0, rIn := ts.reverse },
        ⟨t.rounds, t.k + 2 + (c.kLog - 6) + 2, ByteArray.empty⟩) := by
  obtain ⟨hα, hβ, hR, hlt, hzn, hza, hr⟩ := ha
  unfold holoLincheckT
  dsimp only
  refine (RecAlgebra.bind_of_ok (RecAlgebra.label_eq _ _)).trans ?_
  refine (RecAlgebra.bind_of_ok (sampleOne_of (by exact hα) (by dsimp only; rw [label_enc]; simp [encs]))).trans ?_
  refine (RecAlgebra.bind_of_ok (sampleOne_of (by exact hβ) (by simp [encs]))).trans ?_
  refine (RecAlgebra.bind_of_ok (lcForIn_range_of _ (lStep p.rounds)
    (fun j r => OneCoin r (encs [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1])) ?hb (c.kLog - 6) _ _
    (dot (RecAlgebra.addH comb hpart) p.zPartial, ts) _ (fun j hj => by exact hR j hj) (by exact hfold))).trans ?_
  case hb =>
    intro j s tt hp hq
    obtain ⟨cl, tss⟩ := s
    rcases hg : p.rounds[j]! with ⟨g1, ginf⟩
    dsimp only
    refine (RecAlgebra.bind_of_ok (RecAlgebra.absorbF_eq _ _)).trans ?_
    refine (RecAlgebra.bind_of_ok (RecAlgebra.absorbF_eq _ _)).trans ?_
    refine (RecAlgebra.bind_of_ok (sampleOne_of (by exact hq)
      (by dsimp only; rw [absorbF_enc, absorbF_enc]; simp [encs, hp, hg]))).trans ?_
    unfold lStep
    rw [hg]
    rfl
  dsimp only
  rw [hcomb]
  dsimp only
  refine (RecAlgebra.bind_of_ok (pureOne_of _ _)).trans ?_
  refine (RecAlgebra.bind_of_ok (RecAlgebra.check_eq (by simp [hsize]) _ _)).trans ?_
  refine (RecAlgebra.bind_of_ok (RecAlgebra.absorbSlice_eq _ _)).trans ?_
  refine (RecAlgebra.bind_of_ok (closeEmpty_answered (L := encs [.vec p.zPartial.toList, .sq 2 0]) (by exact hlt)
    (by exact hzn) (by exact hza) (by dsimp only; rw [absorbSlice_enc]; simp [encs]))).trans ?_
  refine (RecAlgebra.bind_of_ok (RecAlgebra.absorbSlice_eq _ _)).trans ?_
  refine (RecAlgebra.bind_of_ok (RecAlgebra.check_eq (by simp) _ _)).trans ?_
  refine (RecAlgebra.bind_of_ok (sampleOne_of (by exact hr) (by dsimp only; rw [absorbSlice_enc]; simp [encs]))).trans ?_
  rfl

open Game FlockLevel3 in
/-- **The lincheck's plays with hidden units from its checks.** Under `holoLincheck_of_checks`' conditions, on a
circuit fold realizing `S` and a zerocheck output that is the model's `zc'`, the model's `holoLincheck`, played with the
proof's lincheck messages and `h` on the coins of the rounds up to `z_partial`'s and of `r'`'s, returns the claim
`holoLincheck_of_checks` hands on, as `ab`, and `h` (`holoLincheck_refines`). -/
theorem holoLincheck_plays_of_checks (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S)
    (zc : Flock.ZcOut) (zc' : Model.ZcOut 𝔽 S.m) (hzc : ZcRel S.m zc zc') (p : LincheckProof) (hpart : Array F128)
    (t : Transcript) (hwf : RoundsWF t.rounds) (ha : HoloLcAnswered t c.kLog p hpart) {ts comb : Array F128}
    (hfold : (List.range' 0 (c.kLog - 6)).foldl (fun s j => lStep p.rounds j s (t.rounds[t.k + 2 + j]!).coins)
        ((t.rounds[t.k]!).coins.getD 0 0 * zc.vA + zc.vB + (t.rounds[t.k + 1]!).coins.getD 0 0, #[]) =
      (dot (RecAlgebra.addH comb hpart) p.zPartial, ts))
    (hcomb : c.folded ((t.rounds[t.k]!).coins.getD 0 0) ((t.rounds[t.k + 1]!).coins.getD 0 0) (skipWeights zc.z)
      (zc.rho.extract 0 (c.kLog - 6)) ts = .ok comb)
    (hsize : comb.size = 64) :
    ∃ out : Model.HoloLcOut 𝔽 S.m (S.kLog - 6), Plays fOf (Model.holoLincheck execArith S zc')
        (holoLcMsgs S.kLog p hpart)
        (consumed t.rounds t.k (t.k + 2 + (S.kLog - 6)) ++
          consumed t.rounds (t.k + 2 + (S.kLog - 6) + 1) (t.k + 2 + (c.kLog - 6) + 2)) (some out) ∧
      ClaimRel S.m (abcClaims S.m S.kLog zc
        { w := dot (skipWeights ((t.rounds[t.k + 2 + (c.kLog - 6) + 1]!).coins.getD 0 0)) p.zPartial,
          rPrime := (t.rounds[t.k + 2 + (c.kLog - 6) + 1]!).coins.getD 0 0, rIn := ts.reverse })[0]! out.ab ∧
      out.h = vecOf 64 hpart :=
  (holoLincheck_refines S c hc zc zc' hzc p hpart t _ _ hwf
    (holoLincheck_of_checks c zc p hpart t ha hfold hcomb hsize)).2.2.2

/-- **The rep's program with hidden units returns when its stages' checks hold**: `prog_of_checks` with the lincheck's
rounds and checks as `holoLincheck_of_checks` takes them, ring switching from round `k₂ = k₁ + 2 + (k_log − 6) + 2`
(after `z_partial`'s round and `r'`'s), and `r'` the first coin of round `k₁ + 2 + (k_log − 6) + 1`. -/
theorem holoProg_of_checks (st : Setup) (sch : Flock.Schedule) (p : Proof) (hpart : Array F128)
    (extras : Array Claim) (t : Transcript) (hm : 14 ≤ st.m)
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
    (hlc : HoloLcAnswered ⟨t.rounds, k₁, ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩
      st.fold.kLog p.lincheck hpart)
    {ts comb : Array F128}
    (hlfold : (List.range' 0 (st.fold.kLog - 6)).foldl
        (fun s j => lStep p.lincheck.rounds j s (t.rounds[k₁ + 2 + j]!).coins)
        ((t.rounds[k₁]!).coins.getD 0 0 * p.zerocheck.finalA + p.zerocheck.finalB +
          (t.rounds[k₁ + 1]!).coins.getD 0 0, #[]) =
      (dot (RecAlgebra.addH comb hpart) p.lincheck.zPartial, ts))
    (hcomb : st.fold.folded ((t.rounds[k₁]!).coins.getD 0 0) ((t.rounds[k₁ + 1]!).coins.getD 0 0) (skipWeights z)
      (rho.extract 0 (st.fold.kLog - 6)) ts = .ok comb)
    (hsize : comb.size = 64)
    -- ring switching, from round `k₂`
    {k₂ : ℕ} (hk₂ : k₂ = k₁ + 2 + (st.fold.kLog - 6) + 2)
    {claims : Array Claim}
    (hcl : claims = Flock.abcClaims st.m st.kLog (zcOf p.zerocheck r z rho)
      (lcOf p.lincheck ((t.rounds[k₁ + 2 + (st.fold.kLog - 6) + 1]!).coins.getD 0 0) ts) ++ extras)
    {rpp : ℕ → Array F128} {gamma : Array F128} (hrs : p.ringSwitches.size = claims.size)
    (hshape : ∀ k < claims.size,
      p.ringSwitches[k]!.size = 128 ∧ claims[k]!.skip.size = 64 ∧ claims[k]!.x.size = st.m - 6)
    (hval : ∀ k < claims.size, dot (wOf claims[k]!) p.ringSwitches[k]! = claims[k]!.value)
    (hR : RsRounds ⟨t.rounds, k₂, ByteArray.empty⟩ p.ringSwitches claims.size rpp gamma)
    -- Ligerito, from round `k₂ + n + 1`
    {t₄ : Transcript}
    (hlg : Flock.ligerito st.merkle sch (rsT claims p.ringSwitches rpp gamma) p.cap p.ligerito
      (Flock.basis (rsSws claims rpp gamma)) ⟨t.rounds, k₂ + claims.size + 1, ByteArray.empty⟩ = .ok ((), t₄)) :
    holoRepProg st sch p hpart extras t = .ok ((), t₄) := by
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
      (lcOf p.lincheck ((t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1]!).coins.getD 0 0) ts) ++
        extras).size ≠ 0 := by
    simp [Flock.abcClaims]
  unfold holoRepProg
  refine Eq.trans (bind_run h₁) ?_
  refine Eq.trans (bind_run (holoLincheck_of_checks st.fold _ p.lincheck hpart _ hlc hlfold hcomb hsize)) ?_
  refine Eq.trans (bind_run (ringSwitch_run st.m _ p.ringSwitches _ rpp gamma hn hrs hshape hval hR)) ?_
  exact hlg

end FlockSoundness.Refine
