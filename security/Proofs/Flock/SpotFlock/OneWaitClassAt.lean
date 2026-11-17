module

public import Proofs.Flock.SpotFlock.FirstHitClassAt
public import Proofs.Flock.SamplingGeneric.TwoOneWaitNumbers
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Masked
import Proofs.Flock.Claims.KS.FindsFlock
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.SpotFlock.RevealsLe

/-! The by-class first-hit bound at a budget at SPOT[Flock]'s code, the finders run on one wait a session. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- At SHA-512 at the deployed relation, with no premise. -/
theorem oneWaitFinds_zk :
    (Claims.SamplingGeneric.Arg.ofFlock
      (FlockDeploy.deployZk _root_.Flock.Readable.relation)).OneWaitFinds FlockDeploy.sha512 :=
  Claims.Flock.finds_of_oneWait FlockDeploy.sha512 (FlockDeploy.deployZk _root_.Flock.Readable.relation)
    (fun _ => FlockDeploy.KS.cap_pos)
    (fun x => (FlockDeploy.KS.k_range_zk _root_.Flock.Readable.relation x).elim
      (fun hk => (FlockDeploy.KS.zk_splits x).imp (fun hsp => ⟨hk, hsp⟩) id) .inr)
    (fun x => FlockDeploy.KS.εtable_nonneg true x)
    (FlockDeploy.KS.Gen.zk_oneWait_untrim (S := FlockDeploy.KS.S₀)) _ (Claims.Sampling.listWithin_deployZk _)

variable (h : Held)

theorem sessWit_reveal_passes_oneWait (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt)
    (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).toCompiled.checked x r)
    (c : (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)).oneWait.EC
      ((SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
        (two h) (rs h) (cap h)).toCompiled.sh x u))
    {w : Claims.Flock.Wit}
    (hw : (((SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).oneWait.toCompiled.sessWit x u r
        (run FlockDeploy.sha512 P (V h) x ω r).seen c).eval FlockDeploy.sha512
          (run FlockDeploy.sha512 P (V h) x ω)).1 = some w) :
    ∀ q ∈ (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen w,
      (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true := by
  have hSess := (sessionsTwo_spot h
    (sessionOpensTwo_of_frameIs h (layoutFor h) (embeds_layoutFor h) (frameIs_built h hB hS))).oneWait
  have hRF : (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).oneWait.toCompiled.RunsFixed FlockDeploy.sha512 := ⟨hSess.fixed, hSess.runs⟩
  obtain ⟨xS, vS, hsess, hacc, hck⟩ := Claims.SamplingGeneric.Compiled.sessWit_some (x := x) hRF P ω r hu c hw
  exact reveal_passes h hB hS hx hP hd hin P ω r (List.mem_of_mem_take hu) hsess hacc hck

theorem code_hwit_oneWait (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) :
    ∀ (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.oneWait.SlotEC x),
      ∀ iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r),
        ∀ hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val,
          ∀ w : Claims.Flock.Wit,
            ((L.oneWait.sessWit x iu.2 r (run FlockDeploy.sha512 P L.V x ω r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval
                FlockDeploy.sha512 (run FlockDeploy.sha512 P L.V x ω)).1 = some w →
              ∀ u' ∈ L.sessBlocks x (L.checked x r) iu.1,
                L.toCompiled.ClassWhole FlockDeploy.sha512 x
                  (Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val)
                  (L.reveal s u' (run FlockDeploy.sha512 P L.V x ω r).seen w) := by
  subst hL
  intro ω r c iu hiu hu w hw u' hu'
  have hi := Claims.SamplingGeneric.getElem?_of_mem_zip_finRange (j := iu.1) (a := iu.2) hiu
  have hsb := Claims.SamplingGeneric.Compiled.sessBlocks_eq_singleton (by intro p q hpq; exact hpq) hi hu.1
  rw [hsb] at hu'
  have hu'' : u' = iu.2 := List.mem_singleton.1 hu'
  rw [hu'']
  intro p hp _ g g' hg hgg
  have hk : gateKey h s Pt d p.2.pos = gateKey h s Pt d g'.val := by
    rw [hg]
    exact (Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg
  exact reveal_key_of_passes h hP hd hin
    (sessWit_reveal_passes_oneWait h hB hS hx hP hd hin P ω r (List.mem_of_getElem? hi) (c iu.1 ⟨iu.2, hu.1⟩) hw)
    p hp g'.val hk

/-- What it still takes: the budget `T` a run, the three counts over the coins, and the linear expected-time bound at
the one-wait compiled verifier's `B_conf,X` and its two by-class finders. -/
theorem code_eventTwoX_oneWait_le_cls_at_budget (hB : Built h) (hS : Addresses h)
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
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      L.εclsOneWait FlockDeploy.sha512 x P T Ns Ks Km := by
  have hA : (Claims.SamplingGeneric.Arg.ofFlock
      (FlockDeploy.deployZk _root_.Flock.Readable.relation)).KS FlockDeploy.sha512 :=
    Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)
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
  have hs : L.parse x = some s := by
    subst hL
    exact hx
  have hPt : L.partition s = some Pt := by
    subst hL
    exact hP
  have hd' : L.two s = some d := by
    subst hL
    exact hd
  have hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u := by
    subst hL hneeds
    exact fun u v w => code_hpos h s u v w
  have hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val := by
    subst hcls
    intro g g' hgg
    exact gateKey_block h ((Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg)
  have hkept : ∀ (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.oneWait.SlotEC x),
      L.oneWait.toCompiled.ClassWhole FlockDeploy.sha512 x cls (L.oneWait.kept FlockDeploy.sha512 P x s ω r c) := by
    subst hcls
    exact fun ω r c => L.oneWait.toCompiled.kept_classWhole_of_wit FlockDeploy.sha512 P x s ω _ r c
      (code_hwit_oneWait h hB hS hL hx hP hd hin P ω r c)
  exact Claims.SamplingGeneric.CompiledTwo.eventTwoX_oneWait_le_cls_at_budget oneWaitFinds_zk hA hbind hSess hopen
    hs hPt hd' hin P needs hpos cls
    (fun a g g' hgg hg =>
      L.oneWait.toCompiled.opensAt_isSome_cls FlockDeploy.sha512 P x a.1 hs cls a.2.1 a.2.2
        (hkept a.1 a.2.1 a.2.2) g g' hgg hg)
    (fun t ec g g' hgg hg =>
      L.oneWait.iopTable_isSome_cls FlockDeploy.sha512 P x t.1 hs d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g'
        hgg hg)
    T hT hNs hKs hKm hBc hX hI

theorem code_eventTwoX_oneWait_le_cls_513 (hB : Built h) (hS : Addresses h)
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
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.finderICls FlockDeploy.sha512 P x s d.part needs cls K))
    (hTle : T ≤ 513 * 2 ^ 80)
    (hk : ∀ u < L.toCompiled.nPU x, (Claims.SamplingGeneric.Arg.ofFlock
      (FlockDeploy.deployZk _root_.Flock.Readable.relation)).k (L.toCompiled.sh x u) ≤ 12343986)
    (hNsle : Ns ≤ 2 ^ 26) (hKsle : Ks ≤ 2 ^ 38) (hKmle : Km ≤ 2 ^ 38)
    (hown : L.toCompiled.nPU x * L.toCompiled.openCalls x ≤ 513 * 2 ^ 80) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB FlockDeploy.sha512 L.V P x) + 2 ^ (-(102 : ℝ)) := by
  have hNs0 : 0 ≤ Ns := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hNs
  have hKs0 : 0 ≤ Ks := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKs
  have hKm0 : 0 ≤ Km := le_trans (Ex_nonneg (by intro r; exact Nat.cast_nonneg _)) hKm
  exact (code_eventTwoX_oneWait_le_cls_at_budget h hB hS hL hx hP hd hin P hneeds hcls T hT hNs hKs hKm hBc hX
    hI).trans (Claims.SamplingGeneric.CompiledTwo.εclsOneWait_le_513 FlockDeploy.sha512 L x P hNs0 hKs0 hKm0 hTle hk
      hNsle hKsle hKmle hown)

end SpotFlock
