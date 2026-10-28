import Proofs.Flock.Claims.Complete.Replay.Lincheck
import Proofs.Flock.Soundness.Refine.OpeningConverse

/-!
# The ring switch replays the honest prover's rounds

`ringSwitchWith_of`: from the honest prover's ring switch (`ringSwitchP`) in the pure session, the verifier's ring switch
(`ringSwitchWith`, PROTOCOL.md §12) replayed on rounds that carry the prover's returns the prover's batched value `T`
and switched claims, consuming exactly the prover's rounds, when each claim's switched vector `s_k` (`switchVec`) meets
its value, `Σ_v w_k[v]·s_k[v] = value_k` (the ring switch's algebra).

* **The prover in closed form** (`ringSwitchP_go`): its ring switches (`Rs.proof`), `T` (`Rs.T`), its switched claims
  (`Rs.sws`) and transcript (`Rs.tEnd`), at the coins of the replies to its sends.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Rs

/-- The coins the prover's ring switch reads: claim `k`'s `r''`, and the batching `γ`. -/
structure Challenges where
  cr : ℕ → ℕ → F128
  cg : ℕ → F128

/-- Claim `k`'s seven coins `r''`. -/
def Challenges.rpp (cs : Challenges) (k : ℕ) : Array F128 := (Array.range 7).map (cs.cr k)

/-- The batching coins `γ`. -/
def gamma (cs : Challenges) (n : ℕ) : Array F128 := (Array.range n).map cs.cg

/-- **The prover's ring switches**: each claim's `s_k`. -/
def proof (tr : Trace) (claims : Array Claim) : Array (Array F128) :=
  pushes (fun k => switchVec tr claims[k]!) claims.size

/-- Each claim's `β_k = Σ_u eq(r''_k, u)·t_k[u]`. -/
def betas (tr : Trace) (claims : Array Claim) (cs : Challenges) : Array F128 :=
  pushes (fun k => dot (eqTable (cs.rpp k)) (FlockSoundness.Refine.tOf (switchVec tr claims[k]!))) claims.size

/-- Each claim's `(r''_k, x_k[1..])`. -/
def perClaim (claims : Array Claim) (cs : Challenges) : Array (Array F128 × Array F128) :=
  pushes (fun k => (cs.rpp k, claims[k]!.x.extract 1 claims[k]!.x.size)) claims.size

/-- **The batched value** `T = Σ_k γ_k·β_k`. -/
def T (tr : Trace) (claims : Array Claim) (cs : Challenges) : F128 := dot (gamma cs claims.size) (betas tr claims cs)

/-- **The switched claims.** -/
def sws (claims : Array Claim) (cs : Challenges) : Array Switched :=
  (Array.range claims.size).map fun k =>
    { gamma := (gamma cs claims.size)[k]!, rpp := ((perClaim claims cs)[k]!).1, z := ((perClaim claims cs)[k]!).2 }

/-- The batch label absorbed. -/
def t₁ (t : Transcript) : Transcript := ⟨t.rounds, t.k, pLabel "flock-pcs-open-batch-v0" t.pending⟩

/-- Claim `k`'s ring-switch label and `s_k` absorbed, from `T`. -/
def absorbed (tr : Trace) (claims : Array Claim) (T : Transcript) (k : ℕ) : Transcript :=
  ⟨T.rounds, T.k, pSlice (switchVec tr claims[k]!) (pLabel "flock-ring-switch-v0" T.pending)⟩

/-- The prover's transcript after `k` claims: each absorbs its label and `s_k` and squeezes its `r''`. -/
def loopT (tr : Trace) (claims : Array Claim) (cs : Challenges) (t : Transcript) : ℕ → Transcript
  | 0 => t₁ t
  | k + 1 => afterSq (absorbed tr claims (loopT tr claims cs t k) k) 2 7 (cs.cr k)

/-- **The prover's transcript after the ring switch**: `γ` squeezed. -/
def tEnd (tr : Trace) (claims : Array Claim) (cs : Challenges) (t : Transcript) : Transcript :=
  afterSq (loopT tr claims cs t claims.size) 2 claims.size cs.cg

/-- The coins of the replies to the sends from the `L`-th on, for `n` claims. -/
def coinsAt (resp : ℕ → Claims.Msg) (L n : ℕ) : Challenges :=
  ⟨fun k => coinOf (resp (L + k)), coinOf (resp (L + n))⟩

end Rs

open Rs

/-- **The prover's ring switch, in closed form**, at the coins of the replies to its sends. -/
theorem ringSwitchP_go (resp : ℕ → Claims.Msg) (tr : Trace) (claims : Array Claim) (hn : claims.size ≠ 0)
    (t : Transcript) (l : Array Claims.Msg) :
    ∃ l', l'.size = l.size + claims.size + 1 ∧
      ((ringSwitchP (asksS resp) tr claims).run t).go l =
        (.ok ((proof tr claims, T tr claims (coinsAt resp l.size claims.size),
            sws claims (coinsAt resp l.size claims.size)),
          tEnd tr claims (coinsAt resp l.size claims.size) t), l') := by
  unfold ringSwitchP
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  dsimp only
  let cs := coinsAt resp l.size claims.size
  let L := loopT tr claims cs t
  refine ⟨?l', ?hsz, ?heq⟩
  case heq =>
    refine Eq.trans (bind_go (forIn_go _
      (fun k => (pushes (fun k => switchVec tr claims[k]!) k,
        pushes (fun k => dot (eqTable (cs.rpp k)) (FlockSoundness.Refine.tOf (switchVec tr claims[k]!))) k,
        pushes (fun k => (cs.rpp k, claims[k]!.x.extract 1 claims[k]!.x.size)) k)) L
      (fun k => Claims.toMsg (framed (absorbed tr claims (L k) k) 2 7))
      l.size ?hbody claims.size _ rfl)) ?tail
    case tail =>
      dsimp only
      rw [bind_go (sampleVecP_go resp hn _ _)]
      rw [pure_go]
      simp only [Array.size_append, List.size_toArray, List.length_map, List.length_range]
      rfl
    case hbody =>
      intro k l₁ hl₁
      dsimp only
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
      rw [bind_go (sampleVecP_go resp (by decide) _ _), hl₁]
      rfl
  case hsz => simp; omega

/-! ## The prover's rounds -/

namespace Rs

variable {tr : Trace} {claims : Array Claim} {cs : Challenges}

theorem loopT_k (t : Transcript) (k : ℕ) : (loopT tr claims cs t k).k = t.k + k := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [loopT, afterSq, absorbed, ih]; omega

theorem loopT_size (t : Transcript) (k : ℕ) : (loopT tr claims cs t k).rounds.size = t.rounds.size + k := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [loopT, afterSq, absorbed, Array.size_push, ih]; omega

theorem loopT_prefix (t : Transcript) (i : ℕ) :
    ∀ j, Prefix (loopT tr claims cs t i).rounds (loopT tr claims cs t (i + j)).rounds
  | 0 => Prefix.refl _
  | j + 1 => (loopT_prefix t i j).trans (Prefix.push _ _)

theorem prefix_loop (t : Transcript) {i : ℕ} (hi : i ≤ claims.size) :
    Prefix (loopT tr claims cs t i).rounds (tEnd tr claims cs t).rounds := by
  have := loopT_prefix (tr := tr) (claims := claims) (cs := cs) t i (claims.size - i)
  rw [show i + (claims.size - i) = claims.size by omega] at this
  exact this.trans (Prefix.push _ _)

theorem tEnd_k (t : Transcript) : (tEnd tr claims cs t).k = t.k + claims.size + 1 := by
  simp only [tEnd, afterSq, loopT_k]

theorem tEnd_size (t : Transcript) : (tEnd tr claims cs t).rounds.size = t.rounds.size + claims.size + 1 := by
  simp only [tEnd, afterSq, Array.size_push, loopT_size]

theorem tEnd_pending (t : Transcript) : (tEnd tr claims cs t).pending = ByteArray.empty := rfl

theorem proof_get (k : ℕ) (hk : k < claims.size) : (proof tr claims)[k]! = switchVec tr claims[k]! :=
  pushes_get _ _ k hk

end Rs

/-- **The verifier's ring switch on the prover's rounds** (§12): from the prover's start `t` (its rounds all recorded,
`t.k = |t.rounds|`), on rounds `R` carrying the prover's from `t.k` on, the ring switch returns the prover's `T` and
switched claims and stops where the prover's transcript did, if each claim's `s_k` meets its value. -/
theorem ringSwitchWith_of (hs : Hashes V) (m : ℕ) (tr : Trace) (claims : Array Claim) (cs : Rs.Challenges)
    (t : Transcript) (hk : t.k = t.rounds.size) (hn : claims.size ≠ 0) {R : Array Round}
    (hR : Agrees R (tEnd tr claims cs t).rounds t.k)
    (hval : ∀ k < claims.size,
      dot (FlockSoundness.Refine.wOf claims[k]!) (switchVec tr claims[k]!) = claims[k]!.value) :
    (ringSwitchWith hs m claims (proof tr claims)).run ⟨R, t.k, t.pending⟩ =
      .ok ((T tr claims cs, sws claims cs), ⟨R, (tEnd tr claims cs t).k, (tEnd tr claims cs t).pending⟩) := by
  have cr : ∀ k < claims.size, Carries R (absorbed tr claims (loopT tr claims cs t k) k).k
      (sent (absorbed tr claims (loopT tr claims cs t k) k) 2 7 (cs.cr k)) := by
    intro k hk'
    have e : (loopT tr claims cs t k).rounds.size = (loopT tr claims cs t k).k := by rw [loopT_size, loopT_k, hk]
    have h := hR.at_push (A := (loopT tr claims cs t k).rounds)
      (x := sent (absorbed tr claims (loopT tr claims cs t k) k) 2 7 (cs.cr k))
      (prefix_loop (i := k + 1) t (by omega)) (by rw [loopT_size, hk]; omega)
    rwa [e] at h
  have cg : Carries R (loopT tr claims cs t claims.size).k
      (sent (loopT tr claims cs t claims.size) 2 claims.size cs.cg) := by
    have e : (loopT tr claims cs t claims.size).rounds.size = (loopT tr claims cs t claims.size).k := by
      rw [loopT_size, loopT_k, hk]
    have h := hR.at_push (A := (loopT tr claims cs t claims.size).rounds)
      (x := sent (loopT tr claims cs t claims.size) 2 claims.size cs.cg) (Prefix.refl _)
      (by rw [loopT_size, hk]; omega)
    rwa [e] at h
  show ringSwitchWith hs m claims (proof tr claims) ⟨R, t.k, t.pending⟩ = _
  unfold ringSwitchWith
  simp only [Flock.With.liftTM_V]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (FlockSoundness.Refine.forIn_range_run _
    (FlockSoundness.Refine.oStep claims (proof tr claims)) cs.rpp
    (fun k => ⟨R, (loopT tr claims cs t k).k, (loopT tr claims cs t k).pending⟩) claims.size _ rfl ?_)) ?_
  · intro i st hi
    obtain ⟨bs, pc⟩ := st
    dsimp only
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)) ?_
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq ?_ _ _)) ?_
    · rw [proof_get i hi]
      exact beq_iff_eq.mpr (hval i hi)
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok
      (sampleVecWith_sent_at hs (by decide) (R := R) (t := absorbed tr claims (loopT tr claims cs t i) i) _ rfl rfl ?_
        (cr i hi))) ?_
    · simp only [absorbed, pSlice, pLabel, proof_get i hi]
    rfl
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleVecWith_sent hs hn cg)) ?_
  obtain ⟨h1, h2⟩ := FlockSoundness.Refine.oStep_fold claims (proof tr claims) cs.rpp (List.range' 0 claims.size)
    (#[], #[])
  have hb : ((List.range' 0 claims.size).foldl
      (fun s i => FlockSoundness.Refine.oStep claims (proof tr claims) i s (cs.rpp i)) (#[], #[])).1 =
        betas tr claims cs := by
    rw [h1, betas, pushes_eq_map, ← List.range_eq_range']
    refine Array.empty_append.trans (congrArg List.toArray (List.map_congr_left fun k hk => ?_))
    rw [proof_get k (List.mem_range.mp hk)]
  have hpc : ((List.range' 0 claims.size).foldl
      (fun s i => FlockSoundness.Refine.oStep claims (proof tr claims) i s (cs.rpp i)) (#[], #[])).2 =
        perClaim claims cs := by
    rw [h2, perClaim, pushes_eq_map, ← List.range_eq_range']
    simp only [Array.empty_append]
  show Except.ok _ = _
  rw [hb, hpc]
  rfl

/-! ## The goal -/

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- The prover's lincheck output at table `i`'s rep `r`. -/
def _root_.FlockComplete.Run.lcOut (o : Run H rel) (i r : ℕ) : LcOut :=
  Lc.out o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.lcCoins i r)

/-- Table `i`'s rep `r`'s transcript after the prover's lincheck. -/
def _root_.FlockComplete.Run.lcEnd (o : Run H rel) (i r : ℕ) : Transcript :=
  Lc.tEnd o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r) (o.zcEnd i r)

/-- The claims table `i`'s rep `r` opens: zerocheck's and lincheck's, then the statement's extra claims. -/
def _root_.FlockComplete.Run.claims (o : Run H rel) (i r : ℕ) : Array Claim :=
  abcClaims o.tabs[i]!.m o.tabs[i]!.kLog (o.zcOut i r) (o.lcOut i r) ++ o.extras i

/-- Table `i`'s rep `r`'s ring-switch coins: the replies from the first send after lincheck's on. -/
def _root_.FlockComplete.Run.rsCoins (o : Run H rel) (i r : ℕ) : Rs.Challenges :=
  Rs.coinsAt o.resp
    (repStart o.spec o.sts (2 * i + r) + 3 + (o.tabs[i]!.m - 6) + 3 + (o.tabs[i]!.fold.kLog - 6))
    (o.claims i r).size

theorem Lc.tEnd_size {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Lc.Challenges}
    (t : Transcript) : (Lc.tEnd c m tr zc full cs t).rounds.size = t.rounds.size + 3 + (c.kLog - 6) := by
  simp only [Lc.tEnd, Lc.t₄, afterSq, Array.size_push, Lc.loopT_size]
  omega

theorem _root_.FlockComplete.Run.lcEnd_k (o : Run H rel) (i r : ℕ) : (o.lcEnd i r).k = (o.lcEnd i r).rounds.size := by
  simp only [Run.lcEnd, Lc.tEnd_k, Lc.tEnd_size, o.zcEnd_k]

/-- **Lincheck's value at a run**: the prover's `LcOut`, and the transcript where the prover's lincheck stopped. -/
theorem _root_.FlockComplete.Run.lc_eq (hR1 : TraceR1CS H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size)
    {r : ℕ} (hr : r < 2) (hzc : ZcAt o i r) (hlc : LcAt o i r) (hal : LcAlgebra H rel) :
    o.lc i r = (o.lcOut i r, ⟨(o.stream i r).rounds, (o.lcEnd i r).k, (o.lcEnd i r).pending⟩) := by
  obtain ⟨hfold, hp, hag⟩ := hlc
  obtain ⟨comb, hcomb, hsize, hcheck⟩ := hal o i hi r hr hfold
  have h := lincheckWith_of o.hs o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r)
    (o.zcEnd i r) (o.zcEnd_k i r) (hag.mono (Nat.zero_le _)) hcomb hsize hcheck
  unfold Run.lc
  rw [o.zc_eq hR1 hi hr hzc, hp]
  exact congrArg val h

/-- **What the session walk establishes for table `i`'s rep `r`'s ring switch**: the decoded proof's ring switches are
the prover's, and the recorded stream carries the prover's rounds through the ring switch. -/
def RsAt (o : Run H rel) (i r : ℕ) : Prop :=
  (o.proof i r).ringSwitches = Rs.proof (o.trace i) (o.claims i r) ∧
    Agrees (o.stream i r).rounds (Rs.tEnd (o.trace i) (o.claims i r) (o.rsCoins i r) (o.lcEnd i r)).rounds 0

/-- **The ring switch's algebra** at every rep: each claim's switched vector of the trace meets its value. -/
def RsAlgebra (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ∀ k < (o.claims i r).size,
    dot (FlockSoundness.Refine.wOf (o.claims i r)[k]!) (switchVec (o.trace i) (o.claims i r)[k]!) =
      (o.claims i r)[k]!.value

theorem _root_.FlockComplete.Run.claims_size_ne (o : Run H rel) (i r : ℕ) : (o.claims i r).size ≠ 0 := by
  simp [Run.claims, abcClaims]

/-- **The ring-switch goal**: the ring switch replays at every honest run, from where lincheck left the transcript, on
R1CS-valid traces, at every rep the session walk aligns, given the lincheck's and the ring switch's algebra. -/
theorem ringSwitch_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r) (hal : LcAlgebra H rel)
    (hrs : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) (hra : RsAlgebra H rel) : RingSwitch H rel := by
  intro o i hi r hr
  obtain ⟨hp, hag⟩ := hrs o i hi r hr
  refine ok_val ?_
  rw [o.lc_eq hR1 hi hr (hzc o i hi r hr) (hlc o i hi r hr) hal, o.zc_eq hR1 hi hr (hzc o i hi r hr), hp]
  exact ⟨_, ringSwitchWith_of o.hs o.tabs[i]!.m (o.trace i) (o.claims i r) (o.rsCoins i r) (o.lcEnd i r)
    (o.lcEnd_k i r) (o.claims_size_ne i r) (hag.mono (Nat.zero_le _)) (hra o i hi r hr)⟩

/-- **A rep that returns** carries the prover's ring switches on its claims, zerocheck's and lincheck's at the replies
from its first send on, where the statement's circuit fold at the lincheck's `α` returns `full`. -/
theorem repP_ringSwitch (resp : ℕ → Claims.Msg) (hsP : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t : Transcript) (l : Array Claims.Msg) {full : Array F128}
    (hfold : st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
      (skipWeights (Zc.out st.m (Zc.coinsAt resp l.size) tr).z)
      ((Zc.out st.m (Zc.coinsAt resp l.size) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full)
    {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repP (asksS resp) hsP st sch tr tbl cm0 extras salt nonce).run t).go l = (.ok (p, t'), l')) :
    p.ringSwitches = Rs.proof tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
      (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
        (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras) := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  obtain ⟨l₂, -, hl⟩ := lincheckP_go resp st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
    (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t) l₁ (by rw [hl₁]; exact hfold)
  rw [← hl₁]
  let claims := abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras
  obtain ⟨l₃, -, hrs⟩ := ringSwitchP_go resp tr claims (by simp [claims, abcClaims])
    (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (Lc.coinsAt resp l₁.size (st.fold.kLog - 6)) (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t))
    l₂
  unfold repP at h
  rw [bind_go hz] at h
  dsimp only at h
  rw [bind_go hl] at h
  dsimp only at h
  rw [bind_go hrs] at h
  refine (?_ : Ret _ fun r : Proof × Transcript => r.1.ringSwitches = Rs.proof tr claims) l₃ (p, t') l' h
  refine Ret.run_bind (Ret.true _) fun lg _ _ => ?_
  rw [StateT.run_pure]
  exact FlockComplete.Ret.pure rfl

end FlockComplete.Replay
