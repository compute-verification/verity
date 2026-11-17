module

public import Proofs.Flock.Claims.ZK.FinishBridge
public import Batteries.Lean.LawfulMonadLift
import Batteries.Lean.LawfulMonad
import Flock.With.SimWalk
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.FinishLig
import Std.Tactic.BVDecide.Normalize.Bool

/-! The live inner rep runs the ZK server's rep but for its stop. -/

@[expose] public section

namespace Flock.ZkServer.Finish

open Flock With Transcript

def liveSim (C : Array (Array F128)) : Sim (StateT ℕ V) LiveM where
  R x y := ∀ k a k', x.run k = .ok (a, k') → k ≤ k' ∧ (k' ≤ C.size → ∀ got : Array (Array F128), got.size = k →
    (∀ i < k, got[i]? = C[i]?) → ∃ got' : Array (Array F128), y.run got = pure (a, got') ∧ got'.size = k' ∧
      ∀ i < k', got'[i]? = C[i]?)
  pure _ _ _ _ h := by
    cases h
    exact ⟨Nat.le_refl _, fun _ got hs hp => ⟨got, rfl, hs, hp⟩⟩
  bind {α β x y f g} hxy hfg k b k'' h := by
    rw [StateT.run_bind] at h
    rcases hx : x.run k with _ | ⟨a, k1⟩
    · rw [hx] at h; cases h
    · rw [hx] at h
      obtain ⟨hk1, hy⟩ := hxy k a k1 hx
      obtain ⟨hk2, hg⟩ := hfg a k1 b k'' h
      refine ⟨hk1.trans hk2, fun hle got hs hp => ?_⟩
      obtain ⟨got1, e1, hs1, hp1⟩ := hy (hk2.trans hle) got hs hp
      obtain ⟨got2, e2, hs2, hp2⟩ := hg hle got1 hs1 hp1
      refine ⟨got2, ?_, hs2, hp2⟩
      rw [StateT.run_bind, e1, pure_bind]
      exact e2
  throw _ _ _ _ h := by cases h
  lift v _ _ _ h := by
    cases v with
    | error e => cases h
    | ok a0 =>
      cases h
      refine ⟨Nat.le_refl _, fun _ got hs hp => ⟨got, ?_, hs, hp⟩⟩
      rw [StateT.run_monadLift]
      exact pure_bind _ _

theorem liveAsks_rel (ch : Frame.Chan) (C C' : Array (Array F128)) (hC : ∀ i < C.size, C'[i]? = C[i]?)
    (live : Bool) : AsksRel (liveSim C) (asksOf C').toC (liveAsks ch C live).toC where
  link _ _ _ _ _ h := by cases h
  round c k a k' h := by
    change ((asksOf C').round c).run k = _ at h
    dsimp only [asksOf] at h
    rw [StateT.run_bind, StateT.run_get, pure_bind] at h
    rcases hk : C'[k]? with _ | cs
    · rw [hk] at h; cases h
    · rw [hk] at h
      dsimp only at h
      split at h
      · rw [StateT.run_bind] at h
        cases h
      · rename_i hn
        rw [StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure] at h
        cases h
        refine ⟨Nat.le_succ k, fun hle got hs hp => ?_⟩
        have hkC : k < C.size := by omega
        have hCk : C[k]? = some cs := (hC k hkC).symm.trans hk
        have hn' := Bool.eq_false_iff.mpr hn
        refine ⟨got.push cs, ?_, by rw [Array.size_push, hs], fun i hi => ?_⟩
        · show ((liveAsks ch C live).round c).run got = _
          simp only [liveAsks, StateT.run_bind, StateT.run_get, StateT.run_set, StateT.run_pure, pure_bind, hs, hCk,
            hn', Bool.false_eq_true, ↓reduceIte]
        · rw [Array.getElem?_push]
          split
          · rename_i hik
            rw [hik, hs]
            exact hCk.symm
          · rename_i hik
            exact hp i (by omega)

section Generic

variable {M N : Type → Type} [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N] [MonadLiftT V M]
  [MonadLiftT V N]

variable (S : Sim M N) {io : Asks M} {io' : Asks N}

theorem zerocheckFw_rel (hio : AsksRel S io.toC io'.toC) (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (mk : ℕ → F128) :
    (S.st Transcript).R (zerocheckFw io m digest cap root lay mk) (zerocheckFw io' m digest cap root lay mk) :=
  zerocheckCut_rel S hio m digest cap root lay mk

theorem lincheckFw_rel (hio : AsksRel S io.toC io'.toC) (c : CircuitFold) (lay : Zk.PadLayout) (mk : ℕ → F128)
    (zc : ZcOut) : (S.st Transcript).R (lincheckFw io c lay mk zc) (lincheckFw io' c lay mk zc) :=
  lincheckCut_rel S hio c 0 lay mk zc

theorem ringSwitchP_rel (hio : AsksRel S io.toC io'.toC) (sv : ℕ → Array F128) (claims : Array Claim) :
    (S.st Transcript).R (ringSwitchP io sv claims) (ringSwitchP io' sv claims) :=
  ringSwitchCut_rel S hio sv claims

theorem ligeritoFw_rel [LawfulMonad M] [LawfulMonad N] (hio : AsksRel S io.toC io'.toC) (sch : Flock.Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (S.st Transcript).R
      (ligeritoPZk io (fun lv g s => pure (commitFw lv g s)) MerkleScheme.hm96Sha512 sch T cm0 t b lay h rep tPad
        extra lanePad salt nonce)
      (ligeritoPZk io' (fun lv g s => pure (commitFw lv g s)) MerkleScheme.hm96Sha512 sch T cm0 t b lay h rep tPad
        extra lanePad salt nonce) := by
  rw [ligeritoPZk_fw, ligeritoPZk_fw]
  exact ligeritoPZk_rel S hio (ofSha512_rel S Sha512.hash) _ _ _ _ _ _ _ _ _ _ _ _ _ _

theorem repPiop_rel (hio : AsksRel S io.toC io'.toC) (st : Setup) (lay : Zk.PadLayout) (cap root : ByteArray)
    (extras : Array Claim) (mk : ℕ → F128) :
    (S.st Transcript).R (repPiop io st lay cap root extras mk) (repPiop io' st lay cap root extras mk) := by
  unfold repPiop
  sim_walk [zerocheckFw_rel S hio, lincheckFw_rel S hio]

theorem repOpening_rel [LawfulMonad M] [LawfulMonad N] (hio : AsksRel S io.toC io'.toC) (st : Setup)
    (hm : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule) (lay : Zk.PadLayout) (tPad rep : ℕ)
    (g : Array F128) (cm0 : Commit) (claims : Array Claim) (h : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) :
    (S.st Transcript).R
      (repOpening io (fun lv g s => pure (commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra lanePad salt)
      (repOpening io' (fun lv g s => pure (commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra lanePad salt) := by
  unfold repOpening
  rw [hm]
  sim_walk [ringSwitchP_rel S hio, ligeritoFw_rel S hio]

theorem repCons_rel [LawfulMonad M] [LawfulMonad N] (hio : AsksRel S io.toC io'.toC) (st : Setup)
    (hm : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    (S.st Transcript).R
      (repCons io (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt)
      (repCons io' (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals extra lanePad
        salt) := by
  unfold repCons
  sim_walk [repPiop_rel S hio, repOpening_rel S hio st hm]

end Generic

theorem repCons_live (ch : Frame.Chan) (C C' : Array (Array F128)) (hC : ∀ i < C.size, C'[i]? = C[i]?) (live : Bool)
    (st : Setup) (hm : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (tr : Transcript)
    {r : Proof × Zk.Level0Msgs × Array (String × Zk.Lin)} {tr' : Transcript} {k' : ℕ}
    (h : ((repCons (asksOf C') (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((r, tr'), k'))
    (hk : k' ≤ C.size) :
    ∃ got : Array (Array F128),
      ((repCons (liveAsks ch C live) (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic
        vals extra lanePad salt).run tr).run #[] = pure ((r, tr'), got) ∧ got.size = k' ∧ ∀ i < k', got[i]? = C[i]? :=
  ((repCons_rel (liveSim C) (liveAsks_rel ch C C' hC live) st hm sch tPad rep dom g cm0 extras pads ic vals extra
    lanePad salt tr 0 (r, tr') k' h).2 hk #[] rfl fun i hi => absurd hi (Nat.not_lt_zero i))

theorem bind_repCons_live (ch : Frame.Chan) (C C' : Array (Array F128)) (hC : ∀ i < C.size, C'[i]? = C[i]?)
    (live : Bool) (st : Setup) (hm : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (tr : Transcript)
    {r : Proof × Zk.Level0Msgs × Array (String × Zk.Lin)} {tr' : Transcript}
    (h : ((repCons (asksOf C') (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((r, tr'), C.size))
    {β : Type} (K : Proof × Zk.Level0Msgs × Array (String × Zk.Lin) → TMW LiveM β) :
    (((repCons (liveAsks ch C live) (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic
      vals extra lanePad salt) >>= K).run tr).run #[] = ((K r).run tr').run C := by
  obtain ⟨got, e, hs, hp⟩ := repCons_live ch C C' hC live st hm sch tPad rep dom g cm0 extras pads ic vals extra
    lanePad salt tr h (Nat.le_refl _)
  have hg : got = C := Array.ext_getElem? fun i => by
    by_cases hi : i < C.size
    · exact hp i hi
    · rw [Array.getElem?_eq_none (by omega), Array.getElem?_eq_none (by omega)]
  subst hg
  rw [StateT.run_bind, StateT.run_bind, e, pure_bind]

theorem sv_bind_ok {α β : Type} {x : StateT ℕ V α} {f : α → StateT ℕ V β} {k : ℕ} {b : β} {k' : ℕ}
    (h : (x >>= f).run k = .ok (b, k')) : ∃ a k₁, x.run k = .ok (a, k₁) ∧ (f a).run k₁ = .ok (b, k') := by
  rw [StateT.run_bind] at h
  rcases hx : x.run k with _ | ⟨a, k₁⟩
  · rw [hx] at h; cases h
  · rw [hx] at h
    exact ⟨a, k₁, rfl, h⟩

theorem tmw_bind_ok {α β : Type} {x : TMW (StateT ℕ V) α} {f : α → TMW (StateT ℕ V) β} {tr : Transcript} {k : ℕ}
    {b : β} {t' : Transcript} {k' : ℕ} (h : ((x >>= f).run tr).run k = .ok ((b, t'), k')) :
    ∃ a tr₁ k₁, (x.run tr).run k = .ok ((a, tr₁), k₁) ∧ ((f a).run tr₁).run k₁ = .ok ((b, t'), k') := by
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, tr₁⟩, k₁, h1, h2⟩ := sv_bind_ok h
  exact ⟨a, tr₁, k₁, h1, h2⟩

theorem repFw_parts (C' : Array (Array F128)) (hs : Hashes (StateT ℕ V)) (st : Setup) (sch : Flock.Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk)
    (ic : InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (tr : Transcript) {pr : Zk.ZkProof} {t' : Transcript} {k : ℕ}
    (h : ((repFw (asksOf C') hs (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
      extra lanePad salt).run tr).run 0 = .ok ((pr, t'), k)) :
    ∃ (p : Proof) (zl : Zk.Level0Msgs) (cons : Array (String × Zk.Lin)) (tr₁ : Transcript) (k₁ : ℕ)
      (inner : Zk.InnerProof),
      ((repCons (asksOf C') (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals extra
        lanePad salt).run tr).run 0 = .ok (((p, zl, cons), tr₁), k₁) ∧
      (failingOn cons pads.h).isEmpty = true ∧
      ((innerP (asksOf C') hs (Zk.PadLayout.new st.m st.kLog sch.k0) pads ic p.zerocheck.finalC cons).run tr₁).run k₁ =
        .ok ((inner, t'), k) ∧
      pr = { proof := p, level0 := zl, inner } := by
  unfold repFw at h
  obtain ⟨⟨p, zl, cons⟩, tr₁, k₁, h1, h2⟩ := tmw_bind_ok h
  dsimp only at h2
  split at h2
  · obtain ⟨u, tr₂, k₂, h3, -⟩ := tmw_bind_ok h2
    cases h3
  · rename_i hne
    obtain ⟨inner, tr₃, k₃, h5, h6⟩ := tmw_bind_ok h2
    cases h6
    exact ⟨p, zl, cons, tr₁, k₁, inner, h1, by simpa using hne, h5, rfl⟩

def innerTail (ch : Frame.Chan) (t : Tab) (q : RepIn) (ic : InnerCommit) (p : Proof)
    (cons : Array (String × Zk.Lin)) (sent : ℕ → ByteArray → IO (Option String))
    (held : ℕ → Array F128 → IO (Option String)) : TMW LiveM Unit := do
  let k := (← getThe (Array (Array F128))).size
  if let some why := consWhy q.coins.size k cons (padsOf q.draws).h then throwThe String why
  let b := consBytes cons
  ch.write ((s!"cons {b.size}\n").toUTF8 ++ b)
  ch.flush
  let _ ← innerP (liveAsks ch q.coins true sent held) (Hashes.ofSha512 Sha512.hash)
    (Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (padsOf q.draws) ic p.zerocheck.finalC cons

def innerProg (ch : Frame.Chan) (t : Tab) (rep : ℕ) (dom : ByteArray) (q : RepIn) (ic : InnerCommit)
    (sent : ℕ → ByteArray → IO (Option String)) (held : ℕ → Array F128 → IO (Option String)) :
    TMW LiveM Unit := do
  let (p, _, cons) ← repCons (liveAsks ch q.coins false) (fun lv g salt => pure (commitFw lv g salt)) t.st t.sch t.tPad
    rep dom t.g t.cm0 (claimsOf q.xs) (padsOf q.draws) ic q.vals t.extra t.lanePad
    (fun l => saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))
  innerTail ch t q ic p cons sent held

theorem innerLive_eq (ch : Frame.Chan) (t : Tab) (rep : ℕ) (dom : ByteArray) (q : RepIn)
    (sent : ℕ → ByteArray → IO (Option String)) (held : ℕ → Array F128 → IO (Option String)) :
    innerLive ch t rep dom q sent held = (do
      let ic ← (monadLift (innerCommitP (Hashes.ofSha512 Sha512.hash) (padsOf q.draws) (saltAt q.draws.salts) :
        V InnerCommit) : LiveM InnerCommit)
      let _ ← (innerProg ch t rep dom q ic sent held).run (Transcript.init #[] dom)) := rfl

theorem innerProg_run (ch : Frame.Chan) (t : Tab) (rep : ℕ) (dom : ByteArray) (q : RepIn) (ic : InnerCommit)
    (sent : ℕ → ByteArray → IO (Option String)) (held : ℕ → Array F128 → IO (Option String))
    (C' : Array (Array F128)) (hC : ∀ i < q.coins.size, C'[i]? = q.coins[i]?)
    (hm : t.st.merkle = MerkleScheme.hm96Sha512) {p : Proof} {zl : Zk.Level0Msgs} {cons : Array (String × Zk.Lin)}
    {tr₁ : Transcript}
    (h : ((repCons (asksOf C') (fun lv g s => pure (commitFw lv g s)) t.st t.sch t.tPad rep dom t.g t.cm0
      (claimsOf q.xs) (padsOf q.draws) ic q.vals t.extra t.lanePad
      (fun l => saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run (Transcript.init #[] dom)).run 0 =
        .ok (((p, zl, cons), tr₁), q.coins.size)) :
    ((innerProg ch t rep dom q ic sent held).run (Transcript.init #[] dom)).run #[] =
      ((innerTail ch t q ic p cons sent held).run tr₁).run q.coins := by
  unfold innerProg
  exact bind_repCons_live ch q.coins C' hC false t.st hm t.sch t.tPad rep dom t.g t.cm0 (claimsOf q.xs) (padsOf q.draws)
    ic q.vals t.extra t.lanePad _ _ h _

end Flock.ZkServer.Finish
