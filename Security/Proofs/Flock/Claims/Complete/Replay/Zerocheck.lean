import Proofs.Flock.Claims.Complete.Replay.Steps

/-!
# Zerocheck replays the honest prover's rounds

`zerocheck_replay`: from the honest prover's zerocheck (`zerocheckP`) in the pure session, the verifier's zerocheck
(`bindAndZerocheckWith`, PROTOCOL.md §9) replayed on rounds that carry the prover's returns the prover's `ZcOut`,
consuming exactly the prover's rounds, when the trace satisfies the block R1CS (`a·b = c` bitwise) and no outer coin is
`1`.

* **The prover in closed form** (`zerocheckP_go`): its rounds (`Zc.rounds`), its proof (`Zc.proof`), its output
  (`Zc.out`), at the coins of the replies to its sends.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Zc

/-- The coins the prover's zerocheck reads: `r_skip`'s, `r_outer`'s, `z`'s, and residual round `i`'s. -/
structure Challenges where
  c0 : ℕ → F128
  c1 : ℕ → F128
  c2 : ℕ → F128
  cr : ℕ → ℕ → F128

/-- The zerocheck vector: the pinned coordinates, then `r_outer`. -/
def Challenges.r (m : ℕ) (cs : Challenges) : Array F128 := pinned ++ (Array.range (m - 13)).map cs.c1

def Challenges.z (cs : Challenges) : F128 := cs.c2 0

/-- Residual round `i`'s challenge. -/
def Challenges.ρ (cs : Challenges) (i : ℕ) : F128 := cs.cr i 0

/-- A bit table's skip row at `z` over `2^n` rows, folded at `ρ 0, …, ρ (i − 1)`, lowest variable first. -/
def tab (n : ℕ) (z : F128) (ρ : ℕ → F128) (bits : Array Bool) : ℕ → Array F128
  | 0 => (Array.range (2 ^ n)).map (skipHat bits (skipWeights z))
  | i + 1 => foldLow (ρ i) (tab n z ρ bits i)

/-- Residual round `i`'s message `(g(1), g(∞))`. -/
def msg (tr : Trace) (n : ℕ) (r : Array F128) (z : F128) (ρ : ℕ → F128) (i : ℕ) : F128 × F128 :=
  let a := tab n z ρ tr.az i
  let b := tab n z ρ tr.bz i
  let h := a.size / 2
  let eqT := eqTable (r.extract (i + 1) n)
  (sumF h fun k => eqT[k]! * (a[2 * k + 1]! * b[2 * k + 1]!),
    sumF h fun k => eqT[k]! * ((a[2 * k]! + a[2 * k + 1]!) * (b[2 * k]! + b[2 * k + 1]!)))

def r1ab (tr : Trace) (n : ℕ) (r : Array F128) : Array F128 :=
  (Array.range 64).map fun i =>
    let w := skipWeights (phi8 (64 + i))
    sumF (2 ^ n) fun x => (eqTable r)[x]! * (skipHat tr.az w x * skipHat tr.bz w x)

def r1c (tr : Trace) (n : ℕ) (r : Array F128) : Array F128 :=
  (Array.range 64).map fun i =>
    let w := skipWeights (phi8 (64 + i))
    sumF (2 ^ n) fun x => (eqTable r)[x]! * skipHat tr.z w x

/-- **The prover's zerocheck proof.** -/
def proof (m : ℕ) (cs : Challenges) (tr : Trace) : ZerocheckProof :=
  ⟨r1ab tr (m - 6) (cs.r m), r1c tr (m - 6) (cs.r m), pushes (msg tr (m - 6) (cs.r m) cs.z cs.ρ) (m - 6),
    (tab (m - 6) cs.z cs.ρ tr.az (m - 6))[0]!, (tab (m - 6) cs.z cs.ρ tr.bz (m - 6))[0]!,
    dot (lagrange nodesL den6Inv cs.z) (r1c tr (m - 6) (cs.r m))⟩

/-- **The prover's `ZcOut`.** -/
def out (m : ℕ) (cs : Challenges) (tr : Trace) : ZcOut :=
  { z := cs.z, rho := pushes cs.ρ (m - 6), rRest := cs.r m, vA := (proof m cs tr).finalA,
    vB := (proof m cs tr).finalB, vC := (proof m cs tr).finalC }

/-- The prover's transcript at the residual rounds, from `t₀`: round `i` absorbs its message and squeezes one coin. -/
def loopT (m : ℕ) (cs : Challenges) (tr : Trace) (t₀ : Transcript) : ℕ → Transcript
  | 0 => t₀
  | i + 1 =>
    let t := loopT m cs tr t₀ i
    let g := msg tr (m - 6) (cs.r m) cs.z cs.ρ i
    afterSq ⟨t.rounds, t.k, pF g.2 (pF g.1 t.pending)⟩ 1 1 (cs.cr i)

/-- The binding's transcript: the statement's digest and the cap absorbed. -/
def t₁ (digest cap : ByteArray) (t : Transcript) : Transcript :=
  ⟨t.rounds, t.k, pLabel "flock-zerocheck-v0" (pBytes cap (pBytes digest (pLabel "flock-r1cs-v0" t.pending)))⟩

def t₂ (digest cap : ByteArray) (cs : Challenges) (t : Transcript) : Transcript := afterSq (t₁ digest cap t) 2 6 cs.c0

def t₃ (m : ℕ) (digest cap : ByteArray) (cs : Challenges) (t : Transcript) : Transcript :=
  afterSq (t₂ digest cap cs t) 2 (m - 13) cs.c1

/-- Round 1's transcript: `round1_ab` and `round1_c` absorbed. -/
def t₄ (m : ℕ) (digest cap : ByteArray) (cs : Challenges) (tr : Trace) (t : Transcript) : Transcript :=
  let t := t₃ m digest cap cs t
  ⟨t.rounds, t.k, pSlice (r1c tr (m - 6) (cs.r m)) (pSlice (r1ab tr (m - 6) (cs.r m)) t.pending)⟩

def t₅ (m : ℕ) (digest cap : ByteArray) (cs : Challenges) (tr : Trace) (t : Transcript) : Transcript :=
  afterSq (t₄ m digest cap cs tr t) 1 1 cs.c2

/-- **The prover's transcript after zerocheck**: the residual rounds, then `final_a`, `final_b` absorbed. -/
def tEnd (m : ℕ) (digest cap : ByteArray) (cs : Challenges) (tr : Trace) (t : Transcript) : Transcript :=
  let t := loopT m cs tr (t₅ m digest cap cs tr t) (m - 6)
  ⟨t.rounds, t.k, pF (proof m cs tr).finalB (pF (proof m cs tr).finalA t.pending)⟩

/-- The coins of the replies to the sends from the `L`-th on. -/
def coinsAt (resp : ℕ → Claims.Msg) (L : ℕ) : Challenges :=
  ⟨coinOf (resp L), coinOf (resp (L + 1)), coinOf (resp (L + 2)), fun i => coinOf (resp (L + 3 + i))⟩

end Zc

open Zc

/-- **The prover's zerocheck, in closed form**, at the coins of the replies to its sends. -/
theorem zerocheckP_go (resp : ℕ → Claims.Msg) {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (tr : Trace)
    (t : Transcript) (l : Array Claims.Msg) :
    ∃ l', l'.size = l.size + 3 + (m - 6) ∧
      ((zerocheckP (asksS resp) m digest cap tr).run t).go l =
        (.ok ((proof m (coinsAt resp l.size) tr, out m (coinsAt resp l.size) tr),
          tEnd m digest cap (coinsAt resp l.size) tr t), l') := by
  unfold zerocheckP
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (sampleVecP_go resp (by decide) _ _)]
  rw [bind_go (sampleVecP_go resp (by omega) _ _)]
  dsimp only
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  dsimp only
  simp only [Array.size_push]
  let cs := coinsAt resp l.size
  let T := loopT m cs tr (t₅ m digest cap cs tr t)
  let g := msg tr (m - 6) (cs.r m) cs.z cs.ρ
  refine ⟨?l', ?hsz, ?heq⟩
  case heq =>
    refine Eq.trans (bind_go (forIn_go _
      (fun i => (tab (m - 6) cs.z cs.ρ tr.az i, tab (m - 6) cs.z cs.ρ tr.bz i, pushes g i, pushes cs.ρ i)) T
      (fun i => Claims.toMsg (framed ⟨(T i).rounds, (T i).k, pF (g i).2 (pF (g i).1 (T i).pending)⟩ 1 1))
      (l.size + 3) ?hbody (m - 6) _ (by simp))) ?tail
    case tail =>
      dsimp only
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
      rw [pure_go]
      rfl
    case hbody =>
      intro i l₁ hl₁
      dsimp only
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
      rw [bind_go (sampleP_go resp _ _), hl₁]
      rfl
  case hsz => simp; omega

/-! ## The prover's rounds -/

namespace Zc

variable {m : ℕ} {digest cap : ByteArray} {cs : Challenges} {tr : Trace}

theorem loopT_k (t₀ : Transcript) (i : ℕ) : (loopT m cs tr t₀ i).k = t₀.k + i := by
  induction i with
  | zero => rfl
  | succ i ih => simp only [loopT, afterSq, ih]; omega

theorem loopT_size (t₀ : Transcript) (i : ℕ) : (loopT m cs tr t₀ i).rounds.size = t₀.rounds.size + i := by
  induction i with
  | zero => rfl
  | succ i ih => simp only [loopT, afterSq, Array.size_push, ih]; omega

theorem loopT_pending (t₀ : Transcript) (h : t₀.pending = ByteArray.empty) (i : ℕ) :
    (loopT m cs tr t₀ i).pending = ByteArray.empty := by
  cases i with
  | zero => exact h
  | succ i => rfl

theorem loopT_prefix (t₀ : Transcript) (i : ℕ) : ∀ j, Prefix (loopT m cs tr t₀ i).rounds (loopT m cs tr t₀ (i + j)).rounds
  | 0 => Prefix.refl _
  | j + 1 => (loopT_prefix t₀ i j).trans (Prefix.push _ _)

theorem tEnd_rounds (t : Transcript) :
    (tEnd m digest cap cs tr t).rounds = (loopT m cs tr (t₅ m digest cap cs tr t) (m - 6)).rounds := rfl

theorem tEnd_k (t : Transcript) : (tEnd m digest cap cs tr t).k = t.k + 3 + (m - 6) := by
  simp only [tEnd, loopT_k, t₅, t₄, t₃, t₂, t₁, afterSq]

theorem tEnd_size (t : Transcript) : (tEnd m digest cap cs tr t).rounds.size = t.rounds.size + 3 + (m - 6) := by
  simp only [tEnd, loopT_size, t₅, t₄, t₃, t₂, t₁, afterSq, Array.size_push]

theorem prefix_t₂ (t : Transcript) : Prefix (t₂ digest cap cs t).rounds (tEnd m digest cap cs tr t).rounds := by
  rw [tEnd_rounds]
  refine Prefix.trans ?_ (by simpa using loopT_prefix (m := m) (cs := cs) (tr := tr) (t₅ m digest cap cs tr t) 0 (m - 6))
  exact (Prefix.push _ _).trans (Prefix.push _ _)

theorem prefix_t₃ (t : Transcript) : Prefix (t₃ m digest cap cs t).rounds (tEnd m digest cap cs tr t).rounds := by
  rw [tEnd_rounds]
  refine Prefix.trans ?_ (by simpa using loopT_prefix (m := m) (cs := cs) (tr := tr) (t₅ m digest cap cs tr t) 0 (m - 6))
  exact Prefix.push _ _

theorem prefix_t₅ (t : Transcript) : Prefix (t₅ m digest cap cs tr t).rounds (tEnd m digest cap cs tr t).rounds := by
  rw [tEnd_rounds]
  have h := loopT_prefix (m := m) (cs := cs) (tr := tr) (t₅ m digest cap cs tr t) 0 (m - 6)
  rw [Nat.zero_add] at h
  exact h

theorem prefix_loop (t : Transcript) {i : ℕ} (hi : i < m - 6) :
    Prefix (loopT m cs tr (t₅ m digest cap cs tr t) (i + 1)).rounds (tEnd m digest cap cs tr t).rounds := by
  rw [tEnd_rounds]
  have := loopT_prefix (m := m) (cs := cs) (tr := tr) (t₅ m digest cap cs tr t) (i + 1) (m - 6 - (i + 1))
  rwa [show i + 1 + (m - 6 - (i + 1)) = m - 6 by omega] at this

/-- Residual round `i`'s retained bytes: its message absorbed from nothing pending, then the squeeze's header. -/
def resBytes (g : F128 × F128) : ByteArray := framed ⟨#[], 0, pF g.2 (pF g.1 ByteArray.empty)⟩ 1 1

/-- `zStep`'s fold, its challenges: each round's first coin. -/
theorem zcFold_snd (r : Array F128) (p : ZerocheckProof) (C : ℕ → Array F128) (claim : F128) (n : ℕ) :
    (Flock.RecAlgebra.zcFold r p C n claim).2 = pushes (fun i => (C i).getD 0 0) n := by
  unfold Flock.RecAlgebra.zcFold
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil]
    simp only [FlockSoundness.Refine.zStep] at ih ⊢
    rw [ih]
    simp [pushes]

end Zc

open Zc

/-- **The verifier's zerocheck on the prover's rounds** (§9): from the prover's start `t` (its rounds all recorded,
`t.k = |t.rounds|`), on rounds `R` carrying the prover's from `t.k` on, the binding and zerocheck return the prover's
`ZcOut` and stop where the prover's transcript did, if the residual rounds' claim ends at `final_a · final_b`. -/
theorem zerocheckWith_of (hs : Hashes V) {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (cs : Zc.Challenges) (tr : Trace)
    (t : Transcript) (hk : t.k = t.rounds.size) {R : Array Round}
    (hR : Agrees R (tEnd m digest cap cs tr t).rounds t.k)
    (hfold : (Flock.RecAlgebra.zcFold (cs.r m) (proof m cs tr) (fun i => #[cs.ρ i]) (m - 6)
      (Flock.RecAlgebra.zcStart (proof m cs tr) cs.z)).1 = (proof m cs tr).finalA * (proof m cs tr).finalB) :
    (bindAndZerocheckWith hs m digest cap (proof m cs tr)).run ⟨R, t.k, t.pending⟩ =
      .ok (out m cs tr, ⟨R, (tEnd m digest cap cs tr t).k, (tEnd m digest cap cs tr t).pending⟩) := by
  have c0 : Carries R (t₁ digest cap t).k (sent (t₁ digest cap t) 2 6 cs.c0) := by
    have h := hR.at_push (A := (t₁ digest cap t).rounds) (x := sent (t₁ digest cap t) 2 6 cs.c0)
      (prefix_t₂ (m := m) (tr := tr) t) (le_of_eq hk)
    rwa [show (t₁ digest cap t).rounds.size = (t₁ digest cap t).k from hk.symm] at h
  have c1 : Carries R (t₂ digest cap cs t).k (sent (t₂ digest cap cs t) 2 (m - 13) cs.c1) := by
    have e : (t₂ digest cap cs t).rounds.size = (t₂ digest cap cs t).k := by
      simp only [t₂, t₁, afterSq, Array.size_push, hk]
    have h := hR.at_push (A := (t₂ digest cap cs t).rounds) (x := sent (t₂ digest cap cs t) 2 (m - 13) cs.c1)
      (prefix_t₃ (tr := tr) t) (by rw [e]; simp only [t₂, t₁, afterSq]; omega)
    rwa [e] at h
  have c2 : Carries R (t₄ m digest cap cs tr t).k (sent (t₄ m digest cap cs tr t) 1 1 cs.c2) := by
    have e : (t₄ m digest cap cs tr t).rounds.size = (t₄ m digest cap cs tr t).k := by
      simp only [t₄, t₃, t₂, t₁, afterSq, Array.size_push, hk]
    have h := hR.at_push (A := (t₄ m digest cap cs tr t).rounds) (x := sent (t₄ m digest cap cs tr t) 1 1 cs.c2)
      (prefix_t₅ t) (by rw [e]; simp only [t₄, t₃, t₂, t₁, afterSq]; omega)
    rwa [e] at h
  have cr : ∀ i < m - 6, Carries R (t.k + 3 + i)
      (sent ⟨(loopT m cs tr (t₅ m digest cap cs tr t) i).rounds, (loopT m cs tr (t₅ m digest cap cs tr t) i).k,
        pF (msg tr (m - 6) (cs.r m) cs.z cs.ρ i).2 (pF (msg tr (m - 6) (cs.r m) cs.z cs.ρ i).1
          (loopT m cs tr (t₅ m digest cap cs tr t) i).pending)⟩ 1 1 (cs.cr i)) := by
    intro i hi
    have e : (loopT m cs tr (t₅ m digest cap cs tr t) i).rounds.size = t.k + 3 + i := by
      rw [loopT_size]
      simp only [t₅, t₄, t₃, t₂, t₁, afterSq, Array.size_push, hk]
      try omega
    have h := hR.at_push (A := (loopT m cs tr (t₅ m digest cap cs tr t) i).rounds)
      (x := sent ⟨(loopT m cs tr (t₅ m digest cap cs tr t) i).rounds, (loopT m cs tr (t₅ m digest cap cs tr t) i).k,
        pF (msg tr (m - 6) (cs.r m) cs.z cs.ρ i).2 (pF (msg tr (m - 6) (cs.r m) cs.z cs.ρ i).1
          (loopT m cs tr (t₅ m digest cap cs tr t) i).pending)⟩ 1 1 (cs.cr i))
      (prefix_loop (digest := digest) (cap := cap) t hi) (by rw [e]; omega)
    rwa [e] at h
  have hrd : ∀ i < m - 6, (R[t.k + 3 + i]!).n = 1 ∧ (R[t.k + 3 + i]!).coins = #[cs.ρ i] ∧
      (R[t.k + 3 + i]!).bytes = some (pF ((proof m cs tr).rounds[i]!).2 (pF ((proof m cs tr).rounds[i]!).1
        ByteArray.empty) ++ header 4 1 1) := by
    intro i hi
    obtain ⟨r, hr, hN, hC, hb⟩ := cr i hi
    have e : R[t.k + 3 + i]! = r := by
      rw [getElem!_def, hr]
    have hp : (proof m cs tr).rounds[i]! = msg tr (m - 6) (cs.r m) cs.z cs.ρ i := pushes_get _ _ i hi
    rw [e, hN, hC, hb, hp]
    refine ⟨rfl, by simp [sent, Challenges.ρ, Array.range_succ_eq_map], ?_⟩
    simp only [sent, framed, loopT_pending (m := m) (cs := cs) (tr := tr) (t₅ m digest cap cs tr t) rfl]
  have hcoin : ∀ i < m - 6, (R[t.k + 3 + i]!).coins.getD 0 0 = (#[cs.ρ i] : Array F128).getD 0 0 := by
    intro i hi
    rw [(hrd i hi).2.1]
  have hfold' := hfold
  rw [Flock.RecAlgebra.zcFold_congr _ _ _ _ (C' := fun i => (R[t.k + 3 + i]!).coins)
    (fun i hi => (hcoin i hi).symm)] at hfold'
  have hrho : (Flock.RecAlgebra.zcFold (cs.r m) (proof m cs tr) (fun i => (R[t.k + 3 + i]!).coins) (m - 6)
      (Flock.RecAlgebra.zcStart (proof m cs tr) cs.z)).2 = pushes cs.ρ (m - 6) := by
    rw [zcFold_snd]
    exact pushes_congr fun i hi => (hcoin i hi).trans rfl
  have hpend : (tEnd m digest cap cs tr t).pending =
      pF (proof m cs tr).finalB (pF (proof m cs tr).finalA ByteArray.empty) := by
    simp only [tEnd, loopT_pending (m := m) (cs := cs) (tr := tr) (t₅ m digest cap cs tr t) rfl]
  rw [tEnd_k, hpend]
  show bindAndZerocheckWith hs m digest cap (proof m cs tr) ⟨R, t.k, t.pending⟩ = _
  unfold bindAndZerocheckWith
  simp only [Flock.With.liftTM_V]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbBytes_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbBytes_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleVecWith_sent hs (by decide) c0)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleVecWith_sent hs (by omega) c1)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at hs _ (by rfl) (by rfl) (by rfl) c2)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (beq_iff_eq.mpr rfl) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.forIn_range_of _
    (FlockSoundness.Refine.zStep (cs.r m) (proof m cs tr).rounds)
    (fun i ρ => ρ.n = 1 ∧ ρ.bytes = some (pF ((proof m cs tr).rounds[i]!).2
      (pF ((proof m cs tr).rounds[i]!).1 ByteArray.empty) ++ header 4 1 1)) ?_ (m - 6)
    (t.k + 3) _ ⟨R, t.k + 3, ByteArray.empty⟩ rfl rfl
    (fun i hi => let h := hrd i (by simp only [List.mem_range'_1] at hi; omega); ⟨h.1, h.2.2⟩))) ?_
  · intro i s tt hp hQ
    rcases hg : (proof m cs tr).rounds[i]! with ⟨g1, ginf⟩
    simp only [hg] at hQ
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_of hs hQ.1 (by
      rw [hQ.2]
      simp only [framed, hp, pF]))) ?_
    simp only [FlockSoundness.Refine.zStep, hg]
    rfl
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (beq_iff_eq.mpr hfold') _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
  show Except.ok (_, _) = Except.ok (_, _)
  congr 2
  simp only [out]
  congr 1

end FlockComplete.Replay
