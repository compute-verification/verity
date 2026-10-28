import Proofs.Flock.Claims.Complete.Zk.InnerMerkle
import Proofs.Flock.Claims.Complete.Zk.Json.Files
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra
import Proofs.Flock.Claims.Complete.Replay.Steps

/-!
# The masked prover's inner proof, and the verifier's inner check on it

* **The prover** (`innerP_go`): the masked prover's inner proof (`innerP`) in the pure session, at the coins of the
  replies to its four sends (`ε`, the batching coins, `β`, the query coins), where its two refusals don't fire: the
  batched constraint holds on the pads, and `β ≠ 0`. Its proof is `InZ.proof` and its transcript `InZ.tEnd`. At a
  rep's pads whose constraints vanish, the first refusal never fires, whatever the coins (`innerP_go_pads`).
* **The verifier** (`verifyInnerWith_of`): on rounds carrying those four sends, from the prover's round and pending
  bytes, `Zk.verifyInnerWith` passes on `InZ.proof` at a rep's pads whose constraints vanish (`ConsHold`), where
  `β ≠ 0` and the hash's digests are 64 bytes: the proof has its sizes (`innerCommitP_root`, `tcOf_size`), the coins are
  the prover's, `τ` opens its commitment, the batched check holds (`innerP_batched`, `innerY_batched`), and each
  query's column is consistent with `Y` (`innerY_column`) and climbs to the pads' root (`innerTree_climb`).
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace InZ

/-- The inner proof's coins, each the reply to its send: `ε`, the batching coins, `β` and the query coins. -/
structure Challenges where
  ce : ℕ → F128
  cg : ℕ → F128
  cb : ℕ → F128
  cq : ℕ → F128

def coinsAt (resp : ℕ → Claims.Msg) (L : ℕ) : Challenges :=
  ⟨coinOf (resp L), coinOf (resp (L + 1)), coinOf (resp (L + 2)), coinOf (resp (L + 3))⟩

variable (lay : Zk.PadLayout) (p : PadsZk) (finalC : F128) (cons : Array (String × Zk.Lin)) (h : ByteArray → ByteArray)
  (ic : InnerCommit) (cs : Challenges)

def rho : F128 := cs.ce 0 * p.h[lay.fa]! + p.h[lay.tx]!

def sigma : F128 := p.h[lay.fb]! + p.h[lay.ty]!

/-- Every constraint and the triple's, at the sacrifice coin. -/
def all : Array (String × Zk.Lin) := cons ++ Zk.triple lay (cs.ce 0) (rho lay p cs) (sigma lay p)

/-- The batched constraint. -/
def lin : Zk.Lin :=
  Zk.Lin.combo ((Array.range (all lay p cons cs).size).map cs.cg) ((all lay p cons cs).map (·.2)) lay.len

def tau : F128 := dot (lin lay p cons cs).a p.mu

/-- `τ`'s commitment through `h`. -/
def tcOf (tau : F128) (salt : ByteArray) : ByteArray := Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes Id) tau salt

def y : Array F128 :=
  (Array.range (p.h ++ p.padH).size).map fun i => (p.h ++ p.padH)[i]! + cs.cb 0 * (p.mu ++ p.padMu)[i]!

def qs : Array ℕ := ((Array.range Zk.Q).map cs.cq).map fun w => w.lo.toNat % ic.ch.size

/-- **The masked prover's inner proof.** -/
def proof : Zk.InnerProof :=
  { root := ic.root, rho := rho lay p cs, sigma := sigma lay p, tauCommit := tcOf h (tau lay p cons cs) p.tauSalt,
    tauSalt := p.tauSalt, tau := tau lay p cons cs, y := y p cs,
    opened := (qs ic cs).map fun j => (ic.ch[j]!, ic.cm[j]!), salts := (qs ic cs).map fun j => ic.salts[j]!,
    paths := (qs ic cs).map fun j =>
      (Array.range (Nat.log2 ic.ch.size)).map fun k => (ic.layers[k]!)[(j >>> k) ^^^ 1]! }

def t₁ (t : Transcript) : Transcript := ⟨t.rounds, t.k, pF finalC (pLabel Zk.LABEL_INNER t.pending)⟩

def t₂ (t : Transcript) : Transcript := afterSq (t₁ finalC t) 1 1 cs.ce

def t₃ (t : Transcript) : Transcript :=
  ⟨(t₂ finalC cs t).rounds, (t₂ finalC cs t).k, pSlice #[rho lay p cs, sigma lay p] (t₂ finalC cs t).pending⟩

def t₄ (t : Transcript) : Transcript := afterSq (t₃ lay p finalC cs t) 2 (all lay p cons cs).size cs.cg

def t₅ (t : Transcript) : Transcript :=
  ⟨(t₄ lay p finalC cons cs t).rounds, (t₄ lay p finalC cons cs t).k,
    pBytes (tcOf h (tau lay p cons cs) p.tauSalt) (t₄ lay p finalC cons cs t).pending⟩

def t₆ (t : Transcript) : Transcript := afterSq (t₅ lay p finalC cons h cs t) 1 1 cs.cb

def t₇ (t : Transcript) : Transcript :=
  ⟨(t₆ lay p finalC cons h cs t).rounds, (t₆ lay p finalC cons h cs t).k,
    pSlice (y p cs) (t₆ lay p finalC cons h cs t).pending⟩

/-- **The masked prover's transcript after its inner proof**: the query coins squeezed. -/
def tEnd (t : Transcript) : Transcript := afterSq (t₇ lay p finalC cons h cs t) 2 Zk.Q cs.cq

end InZ

theorem tauCommitmentWith_ofSha512 (h : ByteArray → ByteArray) (tau : F128) (salt : ByteArray) :
    Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt = pure (InZ.tcOf h tau salt) := rfl

/-- **The masked prover's inner proof, in closed form**, at the coins of the replies to its sends, where the batched
constraint holds on the pads and `β ≠ 0`. -/
theorem innerP_go (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) (t : Transcript) (l : Array Claims.Msg)
    (hz : ((InZ.lin lay p cons (InZ.coinsAt resp l.size)).c +
      dot (InZ.lin lay p cons (InZ.coinsAt resp l.size)).a p.h).isZero = true)
    (hb : (coinOf (resp (l.size + 2)) 0).isZero = false) :
    ∃ l', l'.size = l.size + 4 ∧
      ((innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons).run t).go l =
        (.ok (InZ.proof lay p cons h ic (InZ.coinsAt resp l.size),
          InZ.tEnd lay p finalC cons h (InZ.coinsAt resp l.size) t), l') := by
  unfold innerP
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  dsimp only
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleVecP_go resp (by simp [Zk.triple]) _ _)]
  simp only [InZ.lin, InZ.all, InZ.rho, InZ.sigma, InZ.coinsAt] at hz
  simp only [Array.size_push, hz, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  rw [tauCommitmentWith_ofSha512]
  rw [bind_go (x := StateT.lift (pure _)) rfl]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  simp only [Array.size_push, hb, Bool.false_eq_true, ↓reduceIte]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleVecP_go resp (by decide) _ _)]
  simp only [Array.size_push]
  rw [pure_go]
  exact ⟨_, by simp, rfl⟩

/-- **At a rep's pads, the masked prover's inner proof refuses only at `β = 0`**: where the rep's constraints vanish at
its pads (`ConsHold`), the batched constraint holds at every sacrifice and batching coin (`innerP_batched`), so `innerP`
returns in closed form at every reply but one whose `β` is `0`. -/
theorem innerP_go_pads (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (tape : TapeZk)
    (i r : ℕ) (ic : InnerCommit) (finalC : F128) {cons : Array (String × Zk.Lin)}
    (hc : ConsHold cons (padsOf tape i r lay).h) (t : Transcript) (l : Array Claims.Msg)
    (hb : (coinOf (resp (l.size + 2)) 0).isZero = false) :
    ∃ l', l'.size = l.size + 4 ∧
      ((innerP (asksS resp) (Hashes.ofSha512 h) lay (padsOf tape i r lay) ic finalC cons).run t).go l =
        (.ok (InZ.proof lay (padsOf tape i r lay) cons h ic (InZ.coinsAt resp l.size),
          InZ.tEnd lay (padsOf tape i r lay) finalC cons h (InZ.coinsAt resp l.size) t), l') :=
  innerP_go resp h lay _ ic finalC cons t l
    (innerP_batched tape i r lay hc _ ((Array.range (InZ.all lay (padsOf tape i r lay) cons
      (InZ.coinsAt resp l.size)).size).map (InZ.coinsAt resp l.size).cg)) hb

/-! ## The verifier's inner check on the prover's inner proof -/

theorem bytes_beq_refl (b : ByteArray) : (b == b) = true := by
  show (b.data == b.data) = true
  simp

/-- A loop over `[0:n]` each of whose steps yields and keeps the transcript keeps it. -/
theorem forIn_keep (body : ℕ → PUnit → TM (ForInStep PUnit)) (n : ℕ) (u : Transcript)
    (h : ∀ k < n, body k PUnit.unit u = .ok (.yield PUnit.unit, u)) :
    (forIn [:n] PUnit.unit body : TM PUnit) u = .ok (PUnit.unit, u) := by
  rw [FlockSoundness.Refine.forIn_range_eq]
  suffices ∀ m a, a + m = n → (forIn (List.range' a m) PUnit.unit body : TM PUnit) u = .ok (PUnit.unit, u) from
    this n 0 (by omega)
  intro m
  induction m with
  | zero => intro a _; rfl
  | succ m ih =>
    intro a ha
    rw [FlockSoundness.Refine.forIn_range'_cons, Flock.RecAlgebra.bind_of_ok (h a (by omega))]
    exact ih (a + 1) (by omega)

theorem tcOf_size {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) (tau : F128) (salt : ByteArray) :
    (InZ.tcOf h tau salt).size = 64 :=
  h64 _

/-- The pads' commitment's two rows are the codewords of `h ‖ pad_h` and `μ ‖ pad_μ`. -/
theorem innerCommitP_rows {hs : Hashes V} {p : PadsZk} {salt : ℕ → ByteArray} {ic : InnerCommit}
    (h : innerCommitP hs p salt = .ok ic) :
    ic.ch = (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
      (Zk.codeAt (p.h ++ p.padH)) ∧
    ic.cm = (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
      (Zk.codeAt (p.mu ++ p.padMu)) := by
  unfold innerCommitP at h
  obtain ⟨leaves, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  exact ⟨rfl, rfl⟩

theorem lift_ok {α : Type} {v : V α} {a : α} (h : v = .ok a) (u : Transcript) :
    (monadLift v : TM α) u = .ok (a, u) := by
  subst h
  rfl

theorem map_get {α β : Type} [Inhabited α] [Inhabited β] (a : Array α) (f : α → β) {k : ℕ} (hk : k < a.size) :
    (a.map f)[k]! = f a[k]! := by
  rw [getElem!_pos _ k (by simpa using hk), getElem!_pos a k hk, Array.getElem_map]

theorem qs_get {cq : ℕ → F128} {N k : ℕ} (hk : k < Zk.Q) :
    (((Array.range Zk.Q).map cq).map fun w => w.lo.toNat % N)[k]! = (cq k).lo.toNat % N := by
  rw [getElem!_pos _ k (by simpa using hk)]
  simp

/-- **The verifier's inner check passes on the masked prover's inner proof**, from a transcript at the prover's round
and pending bytes whose rounds carry the prover's four sends: at a rep's pads, where the constraints vanish at the
pads (`ConsHold`), `β ≠ 0` and `h`'s digests are 64 bytes. -/
theorem verifyInnerWith_of (h : ByteArray → ByteArray) (h64 : ∀ b, (h b).size = 64) (lay : Zk.PadLayout)
    (tape : TapeZk) (i r : ℕ) (salt : ℕ → ByteArray) {ic : InnerCommit}
    (hic : innerCommitP (Hashes.ofSha512 h : Hashes V) (padsOf tape i r lay) salt = .ok ic) (finalC : F128)
    {cons : Array (String × Zk.Lin)} (hc : ConsHold cons (padsOf tape i r lay).h) (cs : InZ.Challenges)
    (hb : (cs.cb 0).isZero = false) (t : Transcript) (hk : t.k = t.rounds.size) {R : Array Round}
    (hR : Agrees R (InZ.tEnd lay (padsOf tape i r lay) finalC cons h cs t).rounds t.k) :
    (Zk.verifyInnerWith (Hashes.ofSha512 h : Hashes V) lay (InZ.proof lay (padsOf tape i r lay) cons h ic cs) finalC
      cons) ⟨R, t.k, t.pending⟩ = .ok ((), ⟨R, t.k + 4, ByteArray.empty⟩) := by
  set p := padsOf tape i r lay with hp
  have ht := innerCommitP_tree hic
  have hN : ic.ch.size = (lay.len + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE := by
    rw [ht.ch, padsOf_symbols]
  have c1 : Carries R t.k (sent (InZ.t₁ finalC t) 1 1 cs.ce) := by
    have := hR.at_push (A := (InZ.t₁ finalC t).rounds) (x := sent (InZ.t₁ finalC t) 1 1 cs.ce)
      (((Prefix.push _ _).trans (Prefix.push _ _)).trans (Prefix.push _ _)) (by simp [InZ.t₁, hk])
    rwa [show (InZ.t₁ finalC t).rounds.size = t.k from hk.symm] at this
  have c2 : Carries R (t.k + 1) (sent (InZ.t₃ lay p finalC cs t) 2 (InZ.all lay p cons cs).size cs.cg) := by
    have := hR.at_push (A := (InZ.t₃ lay p finalC cs t).rounds)
      (x := sent (InZ.t₃ lay p finalC cs t) 2 (InZ.all lay p cons cs).size cs.cg)
      ((Prefix.push _ _).trans (Prefix.push _ _)) (by simp [InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk])
    rwa [show (InZ.t₃ lay p finalC cs t).rounds.size = t.k + 1 by simp [InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]] at this
  have c3 : Carries R (t.k + 2) (sent (InZ.t₅ lay p finalC cons h cs t) 1 1 cs.cb) := by
    have := hR.at_push (A := (InZ.t₅ lay p finalC cons h cs t).rounds)
      (x := sent (InZ.t₅ lay p finalC cons h cs t) 1 1 cs.cb)
      (Prefix.push _ _) (by simp [InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]; omega)
    rwa [show (InZ.t₅ lay p finalC cons h cs t).rounds.size = t.k + 2 by
      simp [InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]] at this
  have c4 : Carries R (t.k + 3) (sent (InZ.t₇ lay p finalC cons h cs t) 2 Zk.Q cs.cq) := by
    have := hR.at_push (A := (InZ.t₇ lay p finalC cons h cs t).rounds)
      (x := sent (InZ.t₇ lay p finalC cons h cs t) 2 Zk.Q cs.cq)
      (Prefix.refl _) (by simp [InZ.t₇, InZ.t₆, InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]; omega)
    rwa [show (InZ.t₇ lay p finalC cons h cs t).rounds.size = t.k + 3 by
      simp [InZ.t₇, InZ.t₆, InZ.t₅, InZ.t₄, InZ.t₃, InZ.t₂, InZ.t₁, afterSq, hk]] at this
  have hrows := innerCommitP_rows hic
  have hpow : ic.ch.size = 2 ^ Nat.log2 ic.ch.size := by rw [ht.ch]; exact innerCols_pow _
  unfold Zk.verifyInnerWith
  simp only [Flock.With.liftTM_V]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq ?sz _ _)) ?_
  case sz =>
    simp only [InZ.proof, InZ.y, InZ.qs, Array.size_map, Array.size_range, padsOf_symbols,
      JsonRT.innerCommitP_root h64 _ _ hic, tcOf_size h64, padsOf_tauSalt_size, hp, beq_self_eq_true, Bool.and_self]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.label_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbF_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at _ _ (by rfl) (by rfl) (by rfl) c1)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleVecWith_sent_at _ (by simp [Zk.triple]) _ (by rfl)
    (by rfl) (by rfl) c2)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbBytes_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleWith_sent_at _ _ (by rfl) (by rfl) (by rfl) c3)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (by rw [hb]; rfl) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.absorbSlice_eq _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (sampleVecWith_sent_at _ (by decide) _ (by rfl) (by rfl) (by rfl)
    c4)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok (a := InZ.tcOf h (InZ.tau lay p cons cs) p.tauSalt) rfl _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (bytes_beq_refl _) _ _)) ?_
  have hz := innerP_batched tape i r lay hc (cs.ce 0) ((Array.range (InZ.all lay p cons cs).size).map cs.cg)
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq
    (innerY_batched (InZ.lin lay p cons cs) p.h p.mu p.padH p.padMu (cs.cb 0)
      (by rw [InZ.lin, FlockSoundness.Discharge.Composed.ZerocheckMasked.size_combo, hp, padsOf_size])
      (by rw [hp]; exact padsOf_mu_size tape i r lay) hz) _ _)) ?_
  have ht' : InnerTree (Hashes.ofSha512 h) ic.ch.size ic := by rw [ht.ch]; exact ht
  have hqs : (InZ.qs ic cs).size = Zk.Q := by simp [InZ.qs]
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (forIn_keep _ Zk.Q _ fun k hk' => ?_)) rfl
  have hqk : (InZ.qs ic cs)[k]! = (cs.cq k).lo.toNat % ic.ch.size := qs_get hk'
  have hop : (InZ.proof lay p cons h ic cs).opened[k]! =
      (ic.ch[(cs.cq k).lo.toNat % ic.ch.size]!, ic.cm[(cs.cq k).lo.toNat % ic.ch.size]!) := by
    show ((InZ.qs ic cs).map _)[k]! = _
    rw [map_get _ _ (hqs ▸ hk'), hqk]
  have hsl : (InZ.proof lay p cons h ic cs).salts[k]! = ic.salts[(cs.cq k).lo.toNat % ic.ch.size]! := by
    show ((InZ.qs ic cs).map _)[k]! = _
    rw [map_get _ _ (hqs ▸ hk'), hqk]
  have hpa : (InZ.proof lay p cons h ic cs).paths[k]! = (Array.range (Nat.log2 ic.ch.size)).map
      fun k' => (ic.layers[k']!)[(((cs.cq k).lo.toNat % ic.ch.size) >>> k') ^^^ 1]! := by
    show ((InZ.qs ic cs).map _)[k]! = _
    rw [map_get _ _ (hqs ▸ hk'), hqk]
  rw [qs_get hk', ← hN, hop, hsl, hpa]
  set j := (cs.cq k).lo.toNat % ic.ch.size with hj
  have hjn : j < ic.ch.size := Nat.mod_lt _ (by rw [hpow]; exact Nat.two_pow_pos _)
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq ?col _ _)) ?_
  case col =>
    have hcol := innerY_column (p.h ++ p.padH) (p.mu ++ p.padMu) (cs.cb 0) (by simp [hp, padsOf]) _ _
      (show j < (p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE by rw [← ht.ch]; exact hjn)
    rw [← hrows.1, ← hrows.2] at hcol
    exact hcol
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (by simp) _ _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok rfl _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok (ht'.leaf _ hjn) _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (lift_ok (innerTree_climb ht' hpow hjn) _)) ?_
  refine Eq.trans (Flock.RecAlgebra.bind_of_ok (Flock.RecAlgebra.check_eq (bytes_beq_refl _) _ _)) ?_
  rfl

end FlockComplete.Replay
