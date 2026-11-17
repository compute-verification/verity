module

public import Proofs.Flock.SpotFlock.FirstHitClassAt
public import Proofs.Flock.SamplingGeneric.TwoOffClassAt
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Masked
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.SpotFlock.RevealsLe

/-! The three masses a first-hit error reads, each at most twice a cost in SHA-512 calls over `2^256` at SPOT[Flock]'s
code, where the machine that finds the collision meets the linear expected-time bound. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

theorem code_contract (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)) :
    L.commit.Binds FlockDeploy.sha512 ∧ L.Sessions FlockDeploy.sha512 ∧
      L.toCompiled.OpenCallsLe FlockDeploy.sha512 := by
  have hbind : L.commit.Binds FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.commit_binds (code h) _
  have hSess : L.Sessions FlockDeploy.sha512 := by
    subst hL
    exact sessionsTwo_spot h
      (sessionOpensTwo_of_frameIs h (layoutFor h) (embeds_layoutFor h) (frameIs_built h hB hS))
  have hopen : L.toCompiled.OpenCallsLe FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h)
  exact ⟨hbind, hSess, hopen⟩

theorem code_opens_cls (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) :
    (∀ (a : Ω × ((Fin (L.V.k x) → FlockDeploy.Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt FlockDeploy.sha512 P x a.1 a.2.1 a.2.2 g).isSome →
        (L.opensAt FlockDeploy.sha512 P x a.1 a.2.1 a.2.2 g').isSome) ∧
      ∀ (t : Ω × (Fin (L.V.k x) → FlockDeploy.Coin)) (ec : (Fin (L.V.k x) → FlockDeploy.Coin) × L.SlotEC x)
        (g g' : Fin (L.nG x)), cls g = cls g' →
          (L.iopTable FlockDeploy.sha512 P x t.1 d.part t.2 ec g).isSome →
            (L.iopTable FlockDeploy.sha512 P x t.1 d.part t.2 ec g').isSome := by
  have hs : L.parse x = some s := by
    subst hL
    exact hx
  have hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val := by
    subst hcls
    intro g g' hgg
    exact gateKey_block h ((Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg)
  have hkept : ∀ (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.SlotEC x),
      L.toCompiled.ClassWhole FlockDeploy.sha512 x cls (L.kept FlockDeploy.sha512 P x s ω r c) := by
    subst hcls
    exact fun ω r c => L.toCompiled.kept_classWhole_of_wit FlockDeploy.sha512 P x s ω _ r c
      (code_hwit h hB hS hL hx hP hd hin P ω r c)
  exact ⟨fun a g g' hgg hg =>
      L.toCompiled.opensAt_isSome_cls FlockDeploy.sha512 P x a.1 hs cls a.2.1 a.2.2 (hkept a.1 a.2.1 a.2.2) g g' hgg
        hg,
    fun t ec g g' hgg hg =>
      L.iopTable_isSome_cls FlockDeploy.sha512 P x t.1 hs d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g' hgg hg⟩

theorem code_bconfX_le_at (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P L.V x ω r).pLog.length +
      (run FlockDeploy.sha512 P L.V x ω r).vLog.length ≤ T) {Ns : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x)) :
    (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x).Pr FlockDeploy.sha512 ≤
      2 * L.toCompiled.runCost x T Ns / 2 ^ (256 : ℝ) := by
  obtain ⟨_, hSess, hopen⟩ := code_contract h hB hS hL
  exact Claims.SamplingGeneric.Compiled.bconfX_le_at hSess.toCostContract hopen x P T hT hNs hBc

theorem code_offRegX_le_cls_at (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ}
    (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P L.V x ω r).pLog.length +
      (run FlockDeploy.sha512 P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → FlockDeploy.Coin) × L.SlotEC x) =>
        (run FlockDeploy.sha512 P L.V x q.1 q.2.1).out = .acc ∧
          L.OffRegX FlockDeploy.sha512 P x q.1 q.2.1 q.2.2) ≤
      2 * L.toCompiled.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) := by
  obtain ⟨hbind, hSess, hopen⟩ := code_contract h hB hS hL
  have hs : L.parse x = some s := by
    subst hL
    exact hx
  have hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u := by
    subst hL hneeds
    exact fun u v w => code_hpos h s u v w
  exact Claims.SamplingGeneric.CompiledTwo.pr_acc_offRegX_le_cls_at hbind hSess hopen hs P needs hpos cls
    (code_opens_cls h hB hS hL hx hP hd hin P hcls).1 T hT hNs hKs hKm hX

theorem code_offIreg_le_cls_at (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ}
    (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P L.V x ω r).pLog.length +
      (run FlockDeploy.sha512 P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → FlockDeploy.Coin) × L.SlotEC x) =>
        (run FlockDeploy.sha512 P L.V x q.1 q.2.1).out = .acc ∧
          L.OffIreg FlockDeploy.sha512 P x q.1 d.part q.2.1 q.2.2) ≤
      2 * L.toCompiled.firstHitCost x T Ns Ks Km / 2 ^ (256 : ℝ) := by
  obtain ⟨hbind, hSess, hopen⟩ := code_contract h hB hS hL
  have hs : L.parse x = some s := by
    subst hL
    exact hx
  have hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u := by
    subst hL hneeds
    exact fun u v w => code_hpos h s u v w
  exact Claims.SamplingGeneric.CompiledTwo.pr_acc_offIreg_le_cls_at hbind hSess hopen hs d.part P needs hpos cls
    (code_opens_cls h hB hS hL hx hP hd hin P hcls).2 T hT hNs hKs hKm hI

end SpotFlock
