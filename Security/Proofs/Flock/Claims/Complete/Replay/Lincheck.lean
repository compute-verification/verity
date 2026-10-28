import Proofs.Flock.Claims.Complete.Replay.ZerocheckGoal
import Proofs.Flock.VBridge.Algebra.Lincheck

/-!
# Lincheck replays the honest prover's rounds

`lincheckWith_of`: from the honest prover's lincheck (`lincheckP`) in the pure session, the verifier's lincheck
(`lincheckWith`, PROTOCOL.md §10) replayed on rounds that carry the prover's returns the prover's `LcOut`, consuming
exactly the prover's rounds, when the statement's circuit fold at the prover's coins is some `comb` of width 64 and the
rounds' claim from `α·v_A + v_B + β` ends at `dot comb z_partial` (the lincheck's algebra).

* **The prover in closed form** (`lincheckP_go`): its rounds and `z_partial` (`Lc.proof`), its output (`Lc.out`) and
  transcript (`Lc.tEnd`), at the coins of the replies to its sends, where the circuit fold returns `full`.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Lc

/-- The coins the prover's lincheck reads: `α`'s, `β`'s, round `j`'s, and `r'`'s. -/
structure Challenges where
  ca : ℕ → F128
  cb : ℕ → F128
  cr : ℕ → ℕ → F128
  cf : ℕ → F128

/-- Round `j`'s challenge. -/
def Challenges.τ (cs : Challenges) (j : ℕ) : F128 := cs.cr j 0

/-- The witness's columns folded on the outer variables: `Σ_o eq(ρ_out, o)·z[col + 2^kLog·o]` at `col`. -/
def zr0 (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) : Array F128 :=
  (Array.range (2 ^ c.kLog)).map fun col =>
    sumF (eqTable (zc.rho.extract (c.kLog - 6) (m - 6))).size fun o =>
      if tr.z[col + 2 ^ c.kLog * o]! then (eqTable (zc.rho.extract (c.kLog - 6) (m - 6)))[o]! else 0

/-- The circuit's columns with `β` at the pin, halved at the first `i` challenges. -/
def combT (c : CircuitFold) (full : Array F128) (cs : Challenges) : ℕ → Array F128
  | 0 => full.modify c.pin (· + cs.cb 0)
  | i + 1 => halve (cs.τ i) (combT c full cs i)

/-- The witness's folded columns, halved at the first `i` challenges. -/
def zrT (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (cs : Challenges) : ℕ → Array F128
  | 0 => zr0 c m tr zc
  | i + 1 => halve (cs.τ i) (zrT c m tr zc cs i)

/-- Round `i`'s message `(q(1), q(∞))`. -/
def msg (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128) (cs : Challenges) (i : ℕ) : F128 × F128 :=
  let a := combT c full cs i
  let z := zrT c m tr zc cs i
  (sumF (a.size / 2) fun k => a[k + a.size / 2]! * z[k + a.size / 2]!,
    sumF (a.size / 2) fun k => (a[k]! + a[k + a.size / 2]!) * (z[k]! + z[k + a.size / 2]!))

/-- **The prover's lincheck proof.** -/
def proof (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128) (cs : Challenges) : LincheckProof :=
  ⟨pushes (msg c m tr zc full cs) (c.kLog - 6), zrT c m tr zc cs (c.kLog - 6)⟩

/-- **The prover's `LcOut`.** -/
def out (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (cs : Challenges) : LcOut :=
  { w := dot (skipWeights (cs.cf 0)) (zrT c m tr zc cs (c.kLog - 6)), rPrime := cs.cf 0,
    rIn := (pushes cs.τ (c.kLog - 6)).reverse }

/-- The label absorbed. -/
def t₁ (t : Transcript) : Transcript := ⟨t.rounds, t.k, pLabel "flock-lincheck-v0" t.pending⟩

def t₂ (cs : Challenges) (t : Transcript) : Transcript := afterSq (t₁ t) 1 1 cs.ca

def t₃ (cs : Challenges) (t : Transcript) : Transcript := afterSq (t₂ cs t) 1 1 cs.cb

/-- The prover's transcript at the rounds, from `t₃`: round `i` absorbs its message and squeezes one coin. -/
def loopT (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128) (cs : Challenges) (t : Transcript) :
    ℕ → Transcript
  | 0 => t₃ cs t
  | i + 1 =>
    let T := loopT c m tr zc full cs t i
    let g := msg c m tr zc full cs i
    afterSq ⟨T.rounds, T.k, pF g.2 (pF g.1 T.pending)⟩ 1 1 (cs.cr i)

/-- `z_partial` absorbed. -/
def t₄ (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128) (cs : Challenges) (t : Transcript) :
    Transcript :=
  let T := loopT c m tr zc full cs t (c.kLog - 6)
  ⟨T.rounds, T.k, pSlice (zrT c m tr zc cs (c.kLog - 6)) T.pending⟩

/-- **The prover's transcript after lincheck**: `r'` squeezed. -/
def tEnd (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128) (cs : Challenges) (t : Transcript) :
    Transcript :=
  afterSq (t₄ c m tr zc full cs t) 1 1 cs.cf

/-- The coins of the replies to the sends from the `L`-th on, with `n` rounds. -/
def coinsAt (resp : ℕ → Claims.Msg) (L n : ℕ) : Challenges :=
  ⟨coinOf (resp L), coinOf (resp (L + 1)), fun j => coinOf (resp (L + 2 + j)), coinOf (resp (L + 2 + n))⟩

end Lc

open Lc

/-- **The prover's lincheck, in closed form**, at the coins of the replies to its sends, where the statement's circuit
fold at `α` returns `full`. -/
theorem lincheckP_go (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128)
    (t : Transcript) (l : Array Claims.Msg)
    (hfold : c.fold (coinOf (resp l.size) 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok full) :
    ∃ l', l'.size = l.size + 3 + (c.kLog - 6) ∧
      ((lincheckP (asksS resp) c m tr zc).run t).go l =
        (.ok ((proof c m tr zc full (coinsAt resp l.size (c.kLog - 6)),
            out c m tr zc (coinsAt resp l.size (c.kLog - 6))),
          tEnd c m tr zc full (coinsAt resp l.size (c.kLog - 6)) t), l') := by
  unfold lincheckP
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  rw [bind_go (sampleP_go resp _ _)]
  dsimp only
  simp only [Array.size_push, hfold]
  rw [bind_go (pure_go _ _ _)]
  let cs := coinsAt resp l.size (c.kLog - 6)
  let T := loopT c m tr zc full cs t
  let g := msg c m tr zc full cs
  refine ⟨?l', ?hsz, ?heq⟩
  case heq =>
    refine Eq.trans (bind_go (forIn_go _
      (fun i => (combT c full cs i, zrT c m tr zc cs i, pushes g i, pushes cs.τ i)) T
      (fun i => Claims.toMsg (framed ⟨(T i).rounds, (T i).k, pF (g i).2 (pF (g i).1 (T i).pending)⟩ 1 1))
      (l.size + 2) ?hbody (c.kLog - 6) _ (by simp))) ?tail
    case tail =>
      dsimp only
      rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
      rw [bind_go (sampleP_go resp _ _)]
      rw [pure_go]
      simp only [Array.size_append, Array.size_push, List.size_toArray, List.length_map, List.length_range,
        show l.size + 1 + 1 + (c.kLog - 6) = l.size + 2 + (c.kLog - 6) by omega]
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

namespace Lc

variable {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Challenges}

theorem loopT_k (t : Transcript) (i : ℕ) : (loopT c m tr zc full cs t i).k = t.k + 2 + i := by
  induction i with
  | zero => rfl
  | succ i ih => simp only [loopT, afterSq, ih]; omega

theorem loopT_size (t : Transcript) (i : ℕ) :
    (loopT c m tr zc full cs t i).rounds.size = t.rounds.size + 2 + i := by
  induction i with
  | zero => simp [loopT, t₃, t₂, t₁, afterSq]
  | succ i ih => simp only [loopT, afterSq, Array.size_push, ih]; omega

theorem loopT_pending (t : Transcript) (i : ℕ) : (loopT c m tr zc full cs t i).pending = ByteArray.empty := by
  cases i <;> rfl

theorem loopT_prefix (t : Transcript) (i : ℕ) :
    ∀ j, Prefix (loopT c m tr zc full cs t i).rounds (loopT c m tr zc full cs t (i + j)).rounds
  | 0 => Prefix.refl _
  | j + 1 => (loopT_prefix t i j).trans (Prefix.push _ _)

theorem prefix_loop (t : Transcript) {i : ℕ} (hi : i ≤ c.kLog - 6) :
    Prefix (loopT c m tr zc full cs t i).rounds (tEnd c m tr zc full cs t).rounds := by
  have := loopT_prefix (c := c) (m := m) (tr := tr) (zc := zc) (full := full) (cs := cs) t i (c.kLog - 6 - i)
  rw [show i + (c.kLog - 6 - i) = c.kLog - 6 by omega] at this
  exact this.trans (Prefix.push _ _)

theorem prefix_t₂ (t : Transcript) : Prefix (t₂ cs t).rounds (tEnd c m tr zc full cs t).rounds :=
  (Prefix.push _ _).trans (prefix_loop (i := 0) t (Nat.zero_le _))

theorem tEnd_k (t : Transcript) : (tEnd c m tr zc full cs t).k = t.k + 3 + (c.kLog - 6) := by
  simp only [tEnd, t₄, afterSq, loopT_k]; omega

theorem tEnd_pending (t : Transcript) : (tEnd c m tr zc full cs t).pending = ByteArray.empty := rfl

/-- The lincheck's rounds read only each round's first coin. -/
theorem lcFold_congr (p : LincheckProof) {C C' : ℕ → Array F128} (n : ℕ) (claim : F128)
    (h : ∀ i < n, (C i).getD 0 0 = (C' i).getD 0 0) :
    Flock.RecAlgebra.lcFold p C n claim = Flock.RecAlgebra.lcFold p C' n claim := by
  unfold Flock.RecAlgebra.lcFold
  suffices e : ∀ (l : List ℕ) (s : F128 × Array F128), (∀ i ∈ l, i < n) →
      l.foldl (fun s j => FlockSoundness.Refine.lStep p.rounds j s (C j)) s =
        l.foldl (fun s j => FlockSoundness.Refine.lStep p.rounds j s (C' j)) s from
    e _ _ fun i hi => by simp only [List.mem_range'_1] at hi; omega
  intro l
  induction l with
  | nil => intro s _; rfl
  | cons i l ih =>
      intro s hl
      rw [List.foldl_cons, List.foldl_cons, ih _ fun j hj => hl j (List.mem_cons_of_mem _ hj)]
      congr 1
      simp only [FlockSoundness.Refine.lStep, h i (hl i List.mem_cons_self)]

/-- The lincheck's rounds' challenges: each round's first coin. -/
theorem lcFold_snd (p : LincheckProof) (C : ℕ → Array F128) (claim : F128) (n : ℕ) :
    (Flock.RecAlgebra.lcFold p C n claim).2 = pushes (fun i => (C i).getD 0 0) n := by
  unfold Flock.RecAlgebra.lcFold
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil]
    simp only [FlockSoundness.Refine.lStep] at ih ⊢
    rw [ih]
    simp [pushes]

end Lc

/-- **The verifier's lincheck on the prover's rounds** (§10): from the prover's start `t` (its rounds all recorded,
`t.k = |t.rounds|`), on rounds `R` carrying the prover's from `t.k` on, the lincheck returns the prover's `LcOut` and
stops where the prover's transcript did, if the circuit fold at the prover's coins is `comb` of width 64 and the rounds'
claim from `α·v_A + v_B + β` ends at `dot comb z_partial`. -/
theorem lincheckWith_of (hs : Hashes V) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128)
    (cs : Lc.Challenges) (t : Transcript) (hk : t.k = t.rounds.size) {R : Array Round}
    (hR : Agrees R (tEnd c m tr zc full cs t).rounds t.k) {comb : Array F128}
    (hcomb : c.folded (cs.ca 0) (cs.cb 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6))
      (pushes cs.τ (c.kLog - 6)) = .ok comb)
    (hsize : comb.size = 64)
    (hcheck : (Flock.RecAlgebra.lcFold (proof c m tr zc full cs) (fun j => #[cs.τ j]) (c.kLog - 6)
      (cs.ca 0 * zc.vA + zc.vB + cs.cb 0)).1 = dot comb (proof c m tr zc full cs).zPartial) :
    (lincheckWith hs c zc (proof c m tr zc full cs)).run ⟨R, t.k, t.pending⟩ =
      .ok (out c m tr zc cs, ⟨R, (tEnd c m tr zc full cs t).k, (tEnd c m tr zc full cs t).pending⟩) := by
  set n := c.kLog - 6 with hn
  set p := proof c m tr zc full cs with hp
  have ca : Carries R (t₁ t).k (sent (t₁ t) 1 1 cs.ca) := by
    have h := hR.at_push (A := (t₁ t).rounds) (x := sent (t₁ t) 1 1 cs.ca) (prefix_t₂ t) (le_of_eq hk)
    rwa [show (t₁ t).rounds.size = (t₁ t).k from hk.symm] at h
  have cb : Carries R (t₂ cs t).k (sent (t₂ cs t) 1 1 cs.cb) := by
    have e : (t₂ cs t).rounds.size = (t₂ cs t).k := by simp only [t₂, t₁, afterSq, Array.size_push, hk]
    have h := hR.at_push (A := (t₂ cs t).rounds) (x := sent (t₂ cs t) 1 1 cs.cb)
      (prefix_loop (i := 0) t (Nat.zero_le _)) (by rw [e]; simp only [t₂, t₁, afterSq]; omega)
    rwa [e] at h
  have cr : ∀ i < n, Carries R (t.k + 2 + i)
      (sent ⟨(loopT c m tr zc full cs t i).rounds, (loopT c m tr zc full cs t i).k,
        pF (msg c m tr zc full cs i).2 (pF (msg c m tr zc full cs i).1
          (loopT c m tr zc full cs t i).pending)⟩ 1 1 (cs.cr i)) := by
    intro i hi
    have e : (loopT c m tr zc full cs t i).rounds.size = t.k + 2 + i := by rw [loopT_size, hk]
    have h := hR.at_push (A := (loopT c m tr zc full cs t i).rounds)
      (x := sent ⟨(loopT c m tr zc full cs t i).rounds, (loopT c m tr zc full cs t i).k,
        pF (msg c m tr zc full cs i).2 (pF (msg c m tr zc full cs i).1
          (loopT c m tr zc full cs t i).pending)⟩ 1 1 (cs.cr i))
      (prefix_loop (i := i + 1) t (by omega)) (by rw [e]; omega)
    rwa [e] at h
  have cf : Carries R (t₄ c m tr zc full cs t).k (sent (t₄ c m tr zc full cs t) 1 1 cs.cf) := by
    have e : (t₄ c m tr zc full cs t).rounds.size = (t₄ c m tr zc full cs t).k := by
      simp only [t₄, loopT_size, loopT_k, hk]
    have h := hR.at_push (A := (t₄ c m tr zc full cs t).rounds) (x := sent (t₄ c m tr zc full cs t) 1 1 cs.cf)
      (Prefix.refl _) (by rw [e]; simp only [t₄, loopT_k]; omega)
    rwa [e] at h
  have hrd : ∀ i < n, (R[t.k + 2 + i]!).n = 1 ∧ (R[t.k + 2 + i]!).coins = #[cs.τ i] ∧
      (R[t.k + 2 + i]!).bytes = some (pF (p.rounds[i]!).2 (pF (p.rounds[i]!).1 ByteArray.empty) ++ header 4 1 1) := by
    intro i hi
    obtain ⟨r, hr, hN, hC, hb⟩ := cr i hi
    have e : R[t.k + 2 + i]! = r := by rw [getElem!_def, hr]
    have hpi : p.rounds[i]! = msg c m tr zc full cs i := pushes_get _ _ i hi
    rw [e, hN, hC, hb, hpi]
    refine ⟨rfl, by simp [sent, Challenges.τ, Array.range_succ_eq_map], ?_⟩
    simp only [sent, framed, loopT_pending]
  have hcoin : ∀ i < n, (R[t.k + 2 + i]!).coins.getD 0 0 = (#[cs.τ i] : Array F128).getD 0 0 := by
    intro i hi
    rw [(hrd i hi).2.1]
  have hloop : (List.range' 0 n).foldl (fun s i => FlockSoundness.Refine.lStep p.rounds i s (R[t.k + 2 + i]!).coins)
      (cs.ca 0 * zc.vA + zc.vB + cs.cb 0, #[]) =
      ((Flock.RecAlgebra.lcFold p (fun j => #[cs.τ j]) n (cs.ca 0 * zc.vA + zc.vB + cs.cb 0)).1, pushes cs.τ n) := by
    have e1 := Lc.lcFold_congr p (C := fun i => (R[t.k + 2 + i]!).coins) (C' := fun j => #[cs.τ j]) n
      (cs.ca 0 * zc.vA + zc.vB + cs.cb 0) hcoin
    have e2 := Lc.lcFold_snd p (fun j => #[cs.τ j]) (cs.ca 0 * zc.vA + zc.vB + cs.cb 0) n
    unfold Flock.RecAlgebra.lcFold at e1 e2
    rw [e1]
    refine Prod.ext rfl ?_
    rw [e2]
    rfl
  have hcomb' : c.folded (cs.ca 0) (cs.cb 0) (skipWeights zc.z) (zc.rho.extract 0 n)
      ((List.range' 0 n).foldl (fun s i => FlockSoundness.Refine.lStep p.rounds i s (R[t.k + 2 + i]!).coins)
        (cs.ca 0 * zc.vA + zc.vB + cs.cb 0, #[])).2 = .ok comb := by
    rw [hloop]
    exact hcomb
  show lincheckWith hs c zc p ⟨R, t.k, t.pending⟩ = _
  unfold lincheckWith
  simp only [Flock.With.liftTM_V]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at hs _ (by rfl) (by rfl) (by rfl) ca)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at hs _ (by rfl) (by rfl) (by rfl) cb)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.forIn_range_of _
    (FlockSoundness.Refine.lStep p.rounds)
    (fun i ρ => ρ.n = 1 ∧ ρ.bytes = some (pF (p.rounds[i]!).2 (pF (p.rounds[i]!).1 ByteArray.empty) ++ header 4 1 1))
    ?_ n (t.k + 2) _ ⟨R, t.k + 2, ByteArray.empty⟩ rfl rfl
    (fun i hi => let h := hrd i (by simp only [List.mem_range'_1] at hi; omega); ⟨h.1, h.2.2⟩))) ?_
  · intro i s tt hpd hQ
    obtain ⟨cl, tss⟩ := s
    rcases hg : p.rounds[i]! with ⟨g1, ginf⟩
    simp only [hg] at hQ
    dsimp only
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
    refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_of hs hQ.1 (by
      rw [hQ.2]
      simp only [framed, hpd, pF]))) ?_
    simp only [FlockSoundness.Refine.lStep, hg]
    rfl
  dsimp only
  rw [hcomb']
  dsimp only
  rw [hloop]
  have hk₄ : t.k + 2 + n = (t₄ c m tr zc full cs t).k := by simp only [t₄, loopT_k, hn]
  have hp₄ : (⟨R, t.k + 2 + n, pSlice p.zPartial ByteArray.empty⟩ : Transcript).pending =
      (t₄ c m tr zc full cs t).pending := by
    simp only [t₄, loopT_pending]
    rfl
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (rfl : (pure comb : TM _) _ = .ok (comb, _))) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (by simp [hsize]) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (beq_iff_eq.mpr hcheck.symm) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at hs _ rfl hk₄ hp₄ cf)) ?_
  rfl

/-! ## The goal -/

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- Table `i`'s rep `r`'s transcript after the prover's zerocheck. -/
def _root_.FlockComplete.Run.zcEnd (o : Run H rel) (i r : ℕ) : Transcript :=
  Zc.tEnd o.tabs[i]!.m o.tabs[i]!.digest (flat (o.proof i r).cap) (o.zcCoins i r) (o.trace i)
    (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)

/-- The prover's zerocheck output at table `i`'s rep `r`. -/
def _root_.FlockComplete.Run.zcOut (o : Run H rel) (i r : ℕ) : ZcOut :=
  Zc.out o.tabs[i]!.m (o.zcCoins i r) (o.trace i)

/-- Table `i`'s rep `r`'s lincheck coins: the replies from the first send after zerocheck's on. -/
def _root_.FlockComplete.Run.lcCoins (o : Run H rel) (i r : ℕ) : Lc.Challenges :=
  Lc.coinsAt o.resp (repStart o.spec o.sts (2 * i + r) + 3 + (o.tabs[i]!.m - 6)) (o.tabs[i]!.fold.kLog - 6)

/-- The statement's circuit fold at the lincheck's `α`. -/
def _root_.FlockComplete.Run.full (o : Run H rel) (i r : ℕ) : Array F128 :=
  val (o.tabs[i]!.fold.fold ((o.lcCoins i r).ca 0) (skipWeights (o.zcOut i r).z)
    ((o.zcOut i r).rho.extract 0 (o.tabs[i]!.fold.kLog - 6)))

theorem _root_.FlockComplete.Run.zcEnd_k (o : Run H rel) (i r : ℕ) : (o.zcEnd i r).k = (o.zcEnd i r).rounds.size := by
  simp only [Run.zcEnd, Zc.tEnd_k, Zc.tEnd_size]
  rfl

/-- **Zerocheck's value at a run**: the prover's `ZcOut`, and the transcript where the prover's zerocheck stopped. -/
theorem _root_.FlockComplete.Run.zc_eq (hR1 : TraceR1CS H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size)
    {r : ℕ} (hr : r < 2) (hzc : ZcAt o i r) :
    o.zc i r = (o.zcOut i r, ⟨(o.stream i r).rounds, (o.zcEnd i r).k, (o.zcEnd i r).pending⟩) := by
  obtain ⟨-, hm, -⟩ := o.tab_m hi
  have h := zerocheckWith_of o.hs (by omega) o.tabs[i]!.digest (flat (o.proof i r).cap) (o.zcCoins i r) (o.trace i)
    (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8) rfl hzc.2
    (Zc.fold_final (by omega) _ _ (hR1 o i hi) (o.zc_c1 hi hr))
  unfold Run.zc
  rw [hzc.1]
  exact congrArg val h

/-- **What the session walk establishes for table `i`'s rep `r`'s lincheck**: the statement's circuit fold returns at
the lincheck's `α`, the decoded proof's lincheck is the prover's, and the recorded stream carries the prover's rounds
through lincheck. -/
def LcAt (o : Run H rel) (i r : ℕ) : Prop :=
  o.tabs[i]!.fold.fold ((o.lcCoins i r).ca 0) (skipWeights (o.zcOut i r).z)
      ((o.zcOut i r).rho.extract 0 (o.tabs[i]!.fold.kLog - 6)) = .ok (o.full i r) ∧
    (o.proof i r).lincheck =
      Lc.proof o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r) ∧
    Agrees (o.stream i r).rounds
      (Lc.tEnd o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r)
        (o.zcEnd i r)).rounds 0

/-- **The lincheck's algebra** at every rep where the statement's circuit fold at the lincheck's `α` returns: the
statement's folded circuit at the prover's coins is some `comb` of width 64, and the rounds' claim from
`α·v_A + v_B + β`, on the prover's messages, ends at `dot comb z_partial`. -/
def LcAlgebra (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    o.tabs[i]!.fold.fold ((o.lcCoins i r).ca 0) (skipWeights (o.zcOut i r).z)
      ((o.zcOut i r).rho.extract 0 (o.tabs[i]!.fold.kLog - 6)) = .ok (o.full i r) → ∃ comb : Array F128,
    o.tabs[i]!.fold.folded ((o.lcCoins i r).ca 0) ((o.lcCoins i r).cb 0) (skipWeights (o.zcOut i r).z)
      ((o.zcOut i r).rho.extract 0 (o.tabs[i]!.fold.kLog - 6)) (pushes (o.lcCoins i r).τ (o.tabs[i]!.fold.kLog - 6)) =
        .ok comb ∧
    comb.size = 64 ∧
    (Flock.RecAlgebra.lcFold
      (Lc.proof o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r))
      (fun j => #[(o.lcCoins i r).τ j]) (o.tabs[i]!.fold.kLog - 6)
      ((o.lcCoins i r).ca 0 * (o.zcOut i r).vA + (o.zcOut i r).vB + (o.lcCoins i r).cb 0)).1 =
      dot comb (Lc.proof o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r)).zPartial

/-- **The lincheck goal**: lincheck replays at every honest run, from where zerocheck left the transcript, on
R1CS-valid traces, at every rep the session walk aligns, given the lincheck's algebra. -/
theorem lincheck_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r) (hal : LcAlgebra H rel) : Lincheck H rel := by
  intro o i hi r hr
  obtain ⟨hfold, hp, hag⟩ := hlc o i hi r hr
  obtain ⟨comb, hcomb, hsize, hcheck⟩ := hal o i hi r hr hfold
  refine ok_val ?_
  rw [o.zc_eq hR1 hi hr (hzc o i hi r hr), hp]
  exact ⟨_, lincheckWith_of o.hs o.tabs[i]!.fold o.tabs[i]!.m (o.trace i) (o.zcOut i r) (o.full i r) (o.lcCoins i r)
    (o.zcEnd i r) (o.zcEnd_k i r) (hag.mono (Nat.zero_le _)) hcomb hsize hcheck⟩

/-- **A rep that returns** carries the prover's lincheck, at the replies from the first send after zerocheck's on,
where the statement's circuit fold at the lincheck's `α` returns `full`. -/
theorem repP_lincheck (resp : ℕ → Claims.Msg) (hsP : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t : Transcript) (l : Array Claims.Msg) {full : Array F128}
    (hfold : st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
      (skipWeights (Zc.out st.m (Zc.coinsAt resp l.size) tr).z)
      ((Zc.out st.m (Zc.coinsAt resp l.size) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full)
    {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repP (asksS resp) hsP st sch tr tbl cm0 extras salt nonce).run t).go l = (.ok (p, t'), l')) :
    p.lincheck = Lc.proof st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6)) := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  obtain ⟨l₂, -, hl⟩ := lincheckP_go resp st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
    (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t) l₁ (by rw [hl₁]; exact hfold)
  rw [← hl₁]
  unfold repP at h
  rw [bind_go hz] at h
  dsimp only at h
  rw [bind_go hl] at h
  refine (?_ : Ret _ fun r : Proof × Transcript => r.1.lincheck = Lc.proof st.fold st.m tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) l₂ (p, t') l' h
  refine Ret.run_bind (Ret.true _) fun ⟨ss, T, sws⟩ _ _ => ?_
  refine Ret.run_bind (Ret.true _) fun lg _ _ => ?_
  rw [StateT.run_pure]
  exact FlockComplete.Ret.pure rfl

end FlockComplete.Replay
