import Proofs.Flock.Claims.Complete.Replay.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.Piop

/-!
# The masked prover's zerocheck and lincheck in closed form

`zerocheckPZk_go` and `lincheckPZk_go`: the masked prover's zerocheck and lincheck (`zerocheckPZk`, `lincheckPZk`) in
the pure session, at the coins of the replies to its sends. Each message is the plain prover's (`Zc.msg`, `Lc.msg`)
plus its pad, the outputs are the plain prover's (`Zc.out`, `Lc.out`), and the transcripts absorb the padded messages,
zerocheck's binding the pads' root after its label (`ZcZ.tEnd`, `LcZ.tEnd`).
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace ZcZ

open Zc

/-- Residual round `i`'s padded message. -/
def msg (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (n : ℕ) (r : Array F128) (z : F128) (ρ : ℕ → F128)
    (i : ℕ) : F128 × F128 :=
  ((Zc.msg tr n r z ρ i).1 + h[lay.ml i 0]!, (Zc.msg tr n r z ρ i).2 + h[lay.ml i 1]!)

def r1ab (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (n : ℕ) (r : Array F128) : Array F128 :=
  (Array.range 64).map fun i => (Zc.r1ab tr n r)[i]! + h[lay.r1ab i]!

def r1c (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (n : ℕ) (r : Array F128) : Array F128 :=
  (Array.range 64).map fun i => (Zc.r1c tr n r)[i]! + h[lay.r1c i]!

/-- **The masked prover's zerocheck proof**: the plain one's values, each plus its pad. -/
def proof (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (cs : Zc.Challenges) (tr : Trace) : ZerocheckProof :=
  ⟨r1ab lay h tr (m - 6) (cs.r m), r1c lay h tr (m - 6) (cs.r m),
    pushes (msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ) (m - 6),
    (tab (m - 6) cs.z cs.ρ tr.az (m - 6))[0]! + h[lay.fa]!, (tab (m - 6) cs.z cs.ρ tr.bz (m - 6))[0]! + h[lay.fb]!,
    dot (lagrange nodesL den6Inv cs.z) (Zc.r1c tr (m - 6) (cs.r m)) + h[lay.fc]!⟩

/-- The prover's transcript at the residual rounds, from `t₀`. -/
def loopT (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (cs : Zc.Challenges) (tr : Trace) (t₀ : Transcript) : ℕ → Transcript
  | 0 => t₀
  | i + 1 =>
    let t := loopT lay h m cs tr t₀ i
    let g := msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ i
    afterSq ⟨t.rounds, t.k, pF g.2 (pF g.1 t.pending)⟩ 1 1 (cs.cr i)

/-- The binding's transcript: the statement's digest, the cap and the pads' root absorbed. -/
def t₁ (digest cap root : ByteArray) (t : Transcript) : Transcript :=
  ⟨t.rounds, t.k, pBytes root (pLabel Zk.LABEL_PADS (pLabel "flock-zerocheck-v0"
    (pBytes cap (pBytes digest (pLabel "flock-r1cs-v0" t.pending)))))⟩

def t₂ (digest cap root : ByteArray) (cs : Zc.Challenges) (t : Transcript) : Transcript :=
  afterSq (t₁ digest cap root t) 2 6 cs.c0

def t₃ (m : ℕ) (digest cap root : ByteArray) (cs : Zc.Challenges) (t : Transcript) : Transcript :=
  afterSq (t₂ digest cap root cs t) 2 (m - 13) cs.c1

/-- Round 1's transcript: the padded `round1_ab` and `round1_c` absorbed. -/
def t₄ (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (digest cap root : ByteArray) (cs : Zc.Challenges) (tr : Trace)
    (t : Transcript) : Transcript :=
  let t := t₃ m digest cap root cs t
  ⟨t.rounds, t.k, pSlice (r1c lay h tr (m - 6) (cs.r m)) (pSlice (r1ab lay h tr (m - 6) (cs.r m)) t.pending)⟩

def t₅ (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (digest cap root : ByteArray) (cs : Zc.Challenges) (tr : Trace)
    (t : Transcript) : Transcript :=
  afterSq (t₄ lay h m digest cap root cs tr t) 1 1 cs.c2

/-- **The masked prover's transcript after zerocheck**: the residual rounds, then the padded `final_a`, `final_b`. -/
def tEnd (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (digest cap root : ByteArray) (cs : Zc.Challenges) (tr : Trace)
    (t : Transcript) : Transcript :=
  let t := loopT lay h m cs tr (t₅ lay h m digest cap root cs tr t) (m - 6)
  ⟨t.rounds, t.k, pF (proof lay h m cs tr).finalB (pF (proof lay h m cs tr).finalA t.pending)⟩

end ZcZ

namespace LcZ

open Lc

/-- Round `j`'s padded message. -/
def msg (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128)
    (cs : Lc.Challenges) (j : ℕ) : F128 × F128 :=
  ((Lc.msg c m tr zc full cs j).1 + h[lay.lc j 0]!, (Lc.msg c m tr zc full cs j).2 + h[lay.lc j 1]!)

/-- The padded `z_partial`. -/
def zp (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (cs : Lc.Challenges) :
    Array F128 :=
  (Array.range 64).map fun i => (zrT c m tr zc cs (c.kLog - 6))[i]! + h[lay.zp i]!

/-- **The masked prover's lincheck proof.** -/
def proof (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128)
    (cs : Lc.Challenges) : LincheckProof :=
  ⟨pushes (msg lay h c m tr zc full cs) (c.kLog - 6), zp lay h c m tr zc cs⟩

/-- The prover's transcript at the rounds, from `Lc.t₃`. -/
def loopT (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128)
    (cs : Lc.Challenges) (t : Transcript) : ℕ → Transcript
  | 0 => t₃ cs t
  | i + 1 =>
    let T := loopT lay h c m tr zc full cs t i
    let g := msg lay h c m tr zc full cs i
    afterSq ⟨T.rounds, T.k, pF g.2 (pF g.1 T.pending)⟩ 1 1 (cs.cr i)

/-- The padded `z_partial` absorbed. -/
def t₄ (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128)
    (cs : Lc.Challenges) (t : Transcript) : Transcript :=
  let T := loopT lay h c m tr zc full cs t (c.kLog - 6)
  ⟨T.rounds, T.k, pSlice (zp lay h c m tr zc cs) T.pending⟩

/-- **The masked prover's transcript after lincheck**: `r'` squeezed. -/
def tEnd (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) (full : Array F128)
    (cs : Lc.Challenges) (t : Transcript) : Transcript :=
  afterSq (t₄ lay h c m tr zc full cs t) 1 1 cs.cf

end LcZ

open Zc

/-- **The masked prover's zerocheck, in closed form**, at the coins of the replies to its sends. -/
theorem zerocheckPZk_go (resp : ℕ → Claims.Msg) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray)
    (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (t : Transcript) (l : Array Claims.Msg) :
    ∃ l', l'.size = l.size + 3 + (m - 6) ∧
      ((zerocheckPZk (asksS resp) m digest cap root lay h tr).run t).go l =
        (.ok ((ZcZ.proof lay h m (coinsAt resp l.size) tr, out m (coinsAt resp l.size) tr),
          ZcZ.tEnd lay h m digest cap root (coinsAt resp l.size) tr t), l') := by
  unfold zerocheckPZk
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleVecP_go resp (by decide) _ _)]
  rw [bind_go (sampleVecP_go resp (by omega) _ _)]
  dsimp only
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  dsimp only
  simp only [Array.size_push]
  let cs := coinsAt resp l.size
  let T := ZcZ.loopT lay h m cs tr (ZcZ.t₅ lay h m digest cap root cs tr t)
  let g := ZcZ.msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ
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

open Lc

/-- **The masked prover's lincheck, in closed form**, at the coins of the replies to its sends, where the statement's
circuit fold at `α` returns `full`. -/
theorem lincheckPZk_go (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) (zc : ZcOut) (full : Array F128) (t : Transcript) (l : Array Claims.Msg)
    (hfold : c.fold (coinOf (resp l.size) 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok full) :
    ∃ l', l'.size = l.size + 3 + (c.kLog - 6) ∧
      ((lincheckPZk (asksS resp) c m lay h tr zc).run t).go l =
        (.ok ((LcZ.proof lay h c m tr zc full (Lc.coinsAt resp l.size (c.kLog - 6)),
            Lc.out c m tr zc (Lc.coinsAt resp l.size (c.kLog - 6))),
          LcZ.tEnd lay h c m tr zc full (Lc.coinsAt resp l.size (c.kLog - 6)) t), l') := by
  unfold lincheckPZk
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  rw [bind_go (sampleP_go resp _ _)]
  dsimp only
  simp only [Array.size_push, hfold]
  rw [bind_go (pure_go _ _ _)]
  let cs := Lc.coinsAt resp l.size (c.kLog - 6)
  let T := LcZ.loopT lay h c m tr zc full cs t
  let g := LcZ.msg lay h c m tr zc full cs
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

end FlockComplete.Replay
