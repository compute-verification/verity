module

public import Proofs.Flock.SpotFlock.WiredClass
public import Proofs.Flock.SpotFlock.OffClassAt
public import Proofs.Flock.SpotFlock.OneWaitClassAt
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Masked
import Proofs.Flock.Claims.KS.FindsFlock
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# The by-class bounds at a budget, at SPOT[Flock]'s code at the connected program

`OffClassAt` and `OneWaitClassAt` bound the masses a first-hit error reads at SPOT[Flock]'s two-stage code with the
members' programs side by side (`code h`). This file bounds them at the code at the connected program
(`codeConnected h`: the members' programs wired by the code's own wiring map), under `BuiltWired` at that wiring and
`AddressesConnected`, the premises `SpotKSTwo` takes there:

* `needsConnected`, `gateKeyConnected`, `clsConnected`: the gates each proof block's reveal opens there, a gate's key
  there and its class;
* `connected_contract`: the commitment binds, the sessions meet the two-stage contract (`sessionsTwo_wired`) and the
  opening checks call SHA-512 at most `openCalls` times a slot (`revealsLe_wired`);
* `connected_hwit`, `connected_hwit_oneWait`: what an extracted witness reveals holds the classes of the key whole
  (`reveal_key_wired`), at the extraction and at one wait a session;
* `connected_bconfX_le_at`, `connected_offRegX_le_cls_at`, `connected_offIreg_le_cls_at`: the three masses at a
  budget, each where its one machine meets the linear expected-time bound on SHA-512;
* `connected_eventTwoX_oneWait_le_cls_at_budget`: the event of `KSRegTwoX` at the one-wait extraction's committed
  traces, by `CompiledTwo.εclsOneWait`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)

variable (h : Held)

/-- **The gates unit `u`'s proof block needs opened at the connected program**: `needsWired` at the code's own
wiring map, the positions its reveal opens. -/
abbrev needsConnected (s : AuditStatement) (u : ℕ) : Finset ℕ := needsWired h (wireMap h) s u

/-- **A gate's key at the connected program**: `gateKeyWired` at the code's own wiring (the proof blocks that need
it opened, the replay block that holds it and whether it is on that block's boundary in the connected program). -/
noncomputable abbrev gateKeyConnected (s : AuditStatement) (Pt : Partition) (d : Design.Two) (g : ℕ) :
    Finset ℕ × Option ℕ × Bool :=
  gateKeyWired h (wireMap h) (returned h) s Pt d g

/-- **A gate's class at the connected program**, among `n` gates: the representative of the gates with its key
(`gateKeyConnected`). -/
noncomputable def clsConnected (s : AuditStatement) (Pt : Partition) (d : Design.Two) (n : ℕ) : Fin n → Fin n :=
  Verity.FirstHit.clsOf fun g : Fin n => gateKeyConnected h s Pt d g.val

/-- **What the code at the connected program supplies to a bound at a budget**: under `BuiltWired` at the code's own
wiring and `AddressesConnected`, the commitment binds, the sessions meet the two-stage contract, and the opening checks
call SHA-512 at most `openCalls` times a slot. -/
theorem connected_contract (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h)) :
    L.commit.Binds FlockDeploy.sha512 ∧ L.Sessions FlockDeploy.sha512 ∧
      L.toCompiled.OpenCallsLe FlockDeploy.sha512 := by
  subst hL
  exact ⟨SamplingDeploy.commit_binds (codeConnected h) _,
    sessionsTwo_wired h (fun _ _ hd => rewires_of_two hd (wiresFirst_of_two h hd)) (stmtOpens_connected h hB hA),
    SamplingDeploy.openCallsLe_compiled _ (codeConnected h) iroots _ (revealsLe_wired h (wireMap h) (returned h))⟩

/-- **Each opening the code at the connected program reveals is at a gate its proof block needs there.** -/
theorem connected_hpos
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    {s : AuditStatement} {needs : ℕ → Finset ℕ} (hneeds : needs = needsConnected h s) :
    ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u := by
  subst hL hneeds
  exact fun u v w p hp => (mem_revealWired_pos h (wireMap h) (returned h) (f := h.frame) hp).1

/-- **An accepted session's reveal at the connected program holds the classes of the key whole**: `reveal_key_wired`
at the code's own wiring, at a statement the code at the connected program parses, with its partition and its design
there, in the two-stage scope. -/
theorem reveal_key_connected (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parseWired h (wireMap h) (returned h) x = some s)
    (hP : partitionWired h (wireMap h) s = some Pt) (hd : twoWired h (wireMap h) s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ζ : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (code h).checked x r) {xS : Msg} {vS : Seen}
    (hsess : session h u (run FlockDeploy.sha512 P (V h) x ζ r).seen = some (xS, vS)) (hacc : vS.out = .acc)
    {wF : Claims.Flock.Wit}
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true) :
    ∀ p ∈ revealWired h (wireMap h) (returned h) h.frame s u (run FlockDeploy.sha512 P (V h) x ζ r).seen wF,
      ∀ g' : ℕ, gateKeyConnected h s Pt d p.2.pos = gateKeyConnected h s Pt d g' →
        ∃ p' ∈ revealWired h (wireMap h) (returned h) h.frame s u (run FlockDeploy.sha512 P (V h) x ζ r).seen wF,
          p'.1 = p.1 ∧ p'.2.pos = g' ∧ (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 p'.1 p'.2 = true := by
  obtain ⟨s₀, d₀, hx₀, hd₀, hin₀, rfl, rfl, rfl⟩ := wired_stmt h (wireMap h) (returned h) hx hP hd
  exact reveal_key_wired h (wireMap h) (returned h) hB (hA (sourcesNamed_of_two h hd₀))
    (rewires_of_two hd₀ (wiresFirst_of_two h hd₀)) hx₀ hd₀ hin₀ hin.2.2.1 hin.2.2.2.1 P ζ r hu hsess hacc hck

/-- **The by-class bound's hypothesis on what the extracted witnesses reveal, at the connected program**: at the
compiled two-stage verifier `L` at the code there, under `BuiltWired` and `AddressesConnected`, at a statement in the
two-stage scope and every run, for each slot of the run's sample that is its session's first, each witness that
session's extraction returns there and each proof block of the session, what the witness reveals at the block holds
the classes of `gateKeyConnected` whole. -/
theorem connected_hwit (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : L.parse x = some s)
    (hP : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) :
    ∀ (ζ : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.SlotEC x),
      ∀ iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r),
        ∀ hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val,
          ∀ w : Claims.Flock.Wit,
            ((L.sessWit x iu.2 r (run FlockDeploy.sha512 P L.V x ζ r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval
                FlockDeploy.sha512 (run FlockDeploy.sha512 P L.V x ζ)).1 = some w →
              ∀ u' ∈ L.sessBlocks x (L.checked x r) iu.1,
                L.toCompiled.ClassWhole FlockDeploy.sha512 x (clsConnected h s Pt d (L.nG x))
                  (L.reveal s u' (run FlockDeploy.sha512 P L.V x ζ r).seen w) := by
  have hSess : L.Sessions FlockDeploy.sha512 := (connected_contract h hB hA hL).2.1
  subst hL
  have hRF : (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h)).toCompiled.RunsFixed FlockDeploy.sha512 :=
    ⟨hSess.fixed, hSess.runs⟩
  intro ζ r c iu hiu hu w hw u' hu'
  have hi := Claims.SamplingGeneric.getElem?_of_mem_zip_finRange (j := iu.1) (a := iu.2) hiu
  have hsb := Claims.SamplingGeneric.Compiled.sessBlocks_eq_singleton (by intro p q hpq; exact hpq) hi hu.1
  rw [hsb] at hu'
  have hu'' : u' = iu.2 := List.mem_singleton.1 hu'
  rw [hu'']
  have hum := List.mem_of_getElem? hi
  obtain ⟨xS, vS, hsess, hacc, hck⟩ :=
    Claims.SamplingGeneric.Compiled.sessWit_some hRF P ζ r hum (c iu.1 ⟨iu.2, hu.1⟩) hw
  have hu₀ : iu.2 ∈ (code h).checked x r := by
    have hu₁ : iu.2 ∈ (codeWired h (wireMap h) (returned h)).checked x r := List.mem_of_mem_take hum
    rwa [checked_wired h (wireMap h) (returned h) x r] at hu₁
  intro p hp _ g g' hg hgg
  have hk : gateKeyConnected h s Pt d p.2.pos = gateKeyConnected h s Pt d g'.val := by
    rw [hg]
    exact (Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg
  exact reveal_key_connected h hB hA hx hP hd hin P ζ r hu₀ hsess hacc hck p hp g'.val hk

/-- **The by-class bound's hypothesis on what the extracted witnesses reveal, at the connected program, at one
wait**: `connected_hwit` at the one-wait compiled verifier. -/
theorem connected_hwit_oneWait (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : L.parse x = some s)
    (hP : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) :
    ∀ (ζ : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.oneWait.SlotEC x),
      ∀ iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r),
        ∀ hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val,
          ∀ w : Claims.Flock.Wit,
            ((L.oneWait.sessWit x iu.2 r (run FlockDeploy.sha512 P L.V x ζ r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval
                FlockDeploy.sha512 (run FlockDeploy.sha512 P L.V x ζ)).1 = some w →
              ∀ u' ∈ L.sessBlocks x (L.checked x r) iu.1,
                L.toCompiled.ClassWhole FlockDeploy.sha512 x (clsConnected h s Pt d (L.nG x))
                  (L.reveal s u' (run FlockDeploy.sha512 P L.V x ζ r).seen w) := by
  have hSess : L.Sessions FlockDeploy.sha512 := (connected_contract h hB hA hL).2.1
  subst hL
  have hRF : (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h)).oneWait.toCompiled.RunsFixed FlockDeploy.sha512 :=
    ⟨hSess.oneWait.fixed, hSess.oneWait.runs⟩
  intro ζ r c iu hiu hu w hw u' hu'
  have hi := Claims.SamplingGeneric.getElem?_of_mem_zip_finRange (j := iu.1) (a := iu.2) hiu
  have hsb := Claims.SamplingGeneric.Compiled.sessBlocks_eq_singleton (by intro p q hpq; exact hpq) hi hu.1
  rw [hsb] at hu'
  have hu'' : u' = iu.2 := List.mem_singleton.1 hu'
  rw [hu'']
  have hum : iu.2 ∈ (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      (codeConnected h) iroots (twoConnected h) (rsConnected h) (cap h)).toCompiled.checked x r :=
    List.mem_of_getElem? hi
  obtain ⟨xS, vS, hsess, hacc, hck⟩ :=
    Claims.SamplingGeneric.Compiled.sessWit_some (x := x) hRF P ζ r hum (c iu.1 ⟨iu.2, hu.1⟩) hw
  have hu₀ : iu.2 ∈ (code h).checked x r := by
    have hu₁ : iu.2 ∈ (codeWired h (wireMap h) (returned h)).checked x r := List.mem_of_mem_take hum
    rwa [checked_wired h (wireMap h) (returned h) x r] at hu₁
  intro p hp _ g g' hg hgg
  have hk : gateKeyConnected h s Pt d p.2.pos = gateKeyConnected h s Pt d g'.val := by
    rw [hg]
    exact (Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg
  exact reveal_key_connected h hB hA hx hP hd hin P ζ r hu₀ hsess hacc hck p hp g'.val hk

/-- **At the connected program a run's extractions open a class of the key whole or not at all**, under the
statement's roots and under a replay block's interior roots, at the classes of `gateKeyConnected`. -/
theorem connected_opens_cls (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : L.parse x = some s)
    (hP : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {cls : Fin (L.nG x) → Fin (L.nG x)} (hcls : cls = clsConnected h s Pt d (L.nG x)) :
    (∀ (a : Ω × ((Fin (L.V.k x) → FlockDeploy.Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt FlockDeploy.sha512 P x a.1 a.2.1 a.2.2 g).isSome →
        (L.opensAt FlockDeploy.sha512 P x a.1 a.2.1 a.2.2 g').isSome) ∧
      ∀ (t : Ω × (Fin (L.V.k x) → FlockDeploy.Coin)) (ec : (Fin (L.V.k x) → FlockDeploy.Coin) × L.SlotEC x)
        (g g' : Fin (L.nG x)), cls g = cls g' →
          (L.iopTable FlockDeploy.sha512 P x t.1 d.part t.2 ec g).isSome →
            (L.iopTable FlockDeploy.sha512 P x t.1 d.part t.2 ec g').isSome := by
  have hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val := by
    subst hcls
    intro g g' hgg
    exact gateKeyWired_block h (wireMap h) (returned h) ((Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg)
  have hkept : ∀ (ζ : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.SlotEC x),
      L.toCompiled.ClassWhole FlockDeploy.sha512 x cls (L.kept FlockDeploy.sha512 P x s ζ r c) := by
    subst hcls
    exact fun ζ r c => L.toCompiled.kept_classWhole_of_wit FlockDeploy.sha512 P x s ζ _ r c
      (connected_hwit h hB hA hL hx hP hd hin P ζ r c)
  exact ⟨fun a g g' hgg hg =>
      L.toCompiled.opensAt_isSome_cls FlockDeploy.sha512 P x a.1 hx cls a.2.1 a.2.2 (hkept a.1 a.2.1 a.2.2) g g' hgg
        hg,
    fun t ec g g' hgg hg =>
      L.iopTable_isSome_cls FlockDeploy.sha512 P x t.1 hx d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g' hgg hg⟩

/-- **`B_conf,X` finds a collision with probability at most `2·runCost/2^256`, at the connected program**: under
`BuiltWired` and `AddressesConnected`, where every run calls SHA-512 at most `T` times, both parties' calls counted
(`hT`), `N_s` is at least the expected number of sessions in a run's sample and `B_conf,X` meets the linear
expected-time bound. -/
theorem connected_bconfX_le_at (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P L.V x ω r).pLog.length +
      (run FlockDeploy.sha512 P L.V x ω r).vLog.length ≤ T) {Ns : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x)) :
    (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x).Pr FlockDeploy.sha512 ≤
      2 * L.toCompiled.runCost x T Ns / 2 ^ (256 : ℝ) := by
  obtain ⟨_, hSess, hopen⟩ := connected_contract h hB hA hL
  exact Claims.SamplingGeneric.Compiled.bconfX_le_at hSess.toCostContract hopen x P T hT hNs hBc

/-- **An accepted run is off the boundary trace read at the statement with probability at most
`2·firstHitCost/2^256`, at the connected program**: under `BuiltWired` and `AddressesConnected` at a statement in the
two-stage scope, with `needs` the gates each proof block's reveal opens there and `cls` the classes of
`gateKeyConnected`, where every run calls SHA-512 at most `T` times (`hT`), `N_s`, `K_s` and `K_m` bound the three
expected counts, and the by-class finder under the statement's roots meets the linear expected-time bound at every
`K`. -/
theorem connected_offRegX_le_cls_at (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : L.parse x = some s)
    (hP : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ} (hneeds : needs = needsConnected h s)
    {cls : Fin (L.nG x) → Fin (L.nG x)} (hcls : cls = clsConnected h s Pt d (L.nG x)) (T : ℕ)
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
  obtain ⟨hbind, hSess, hopen⟩ := connected_contract h hB hA hL
  exact Claims.SamplingGeneric.CompiledTwo.pr_acc_offRegX_le_cls_at hbind hSess hopen hx P needs
    (connected_hpos h hL hneeds) cls (connected_opens_cls h hB hA hL hx hP hd hin P hcls).1 T hT hNs hKs hKm hX

/-- **An accepted run is off a committed interior with probability at most `2·firstHitCost/2^256`, at the connected
program**: under `connected_offRegX_le_cls_at`'s hypotheses, with the by-class finder under the interior roots of the
design's second-stage partition in place of the one under the statement's roots. -/
theorem connected_offIreg_le_cls_at (hB : BuiltWired h (wireMap h) (returned h)) (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : L.parse x = some s)
    (hP : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ} (hneeds : needs = needsConnected h s)
    {cls : Fin (L.nG x) → Fin (L.nG x)} (hcls : cls = clsConnected h s Pt d (L.nG x)) (T : ℕ)
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
  obtain ⟨hbind, hSess, hopen⟩ := connected_contract h hB hA hL
  exact Claims.SamplingGeneric.CompiledTwo.pr_acc_offIreg_le_cls_at hbind hSess hopen hx d.part P needs
    (connected_hpos h hL hneeds) cls (connected_opens_cls h hB hA hL hx hP hd hin P hcls).2 T hT hNs hKs hKm hI

/-- **The by-class first-hit bound at a budget, the finders run on one wait a session, at the connected program**:
`CompiledTwo.eventTwoX_oneWait_le_cls_at_budget` at the compiled two-stage verifier `L` at the code there, with
`needs` the gates each proof block's reveal opens there and `cls` the classes of `gateKeyConnected`, under
`BuiltWired` and `AddressesConnected` at a statement in the two-stage scope. Where every run calls SHA-512 at most
`T` times (`hT`), `N_s`, `K_s` and `K_m` bound the three expected counts, and the one-wait compiled verifier's
`B_conf,X` and its two by-class finders at every `K` meet the linear expected-time bound, an accepted run samples a
proof block of the one-wait compiled verifier's `FregTwoX` with probability at most `εclsOneWait`. -/
theorem connected_eventTwoX_oneWait_le_cls_at_budget (hB : BuiltWired h (wireMap h) (returned h))
    (hA : AddressesConnected h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeConnected h)
      iroots (twoConnected h) (rsConnected h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : L.parse x = some s)
    (hP : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ} (hneeds : needs = needsConnected h s)
    {cls : Fin (L.nG x) → Fin (L.nG x)} (hcls : cls = clsConnected h s Pt d (L.nG x)) (T : ℕ)
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
  have hKS : (Claims.SamplingGeneric.Arg.ofFlock
      (FlockDeploy.deployZk _root_.Flock.Readable.relation)).KS FlockDeploy.sha512 :=
    Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)
  obtain ⟨hbind, hSess, hopen⟩ := connected_contract h hB hA hL
  have hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val := by
    subst hcls
    intro g g' hgg
    exact gateKeyWired_block h (wireMap h) (returned h) ((Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg)
  have hkept : ∀ (ζ : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.oneWait.SlotEC x),
      L.oneWait.toCompiled.ClassWhole FlockDeploy.sha512 x cls (L.oneWait.kept FlockDeploy.sha512 P x s ζ r c) := by
    subst hcls
    exact fun ζ r c => L.oneWait.toCompiled.kept_classWhole_of_wit FlockDeploy.sha512 P x s ζ _ r c
      (connected_hwit_oneWait h hB hA hL hx hP hd hin P ζ r c)
  exact Claims.SamplingGeneric.CompiledTwo.eventTwoX_oneWait_le_cls_at_budget oneWaitFinds_zk hKS hbind hSess hopen
    hx hP hd hin P needs (connected_hpos h hL hneeds) cls
    (fun a g g' hgg hg =>
      L.oneWait.toCompiled.opensAt_isSome_cls FlockDeploy.sha512 P x a.1 hx cls a.2.1 a.2.2
        (hkept a.1 a.2.1 a.2.2) g g' hgg hg)
    (fun t ec g g' hgg hg =>
      L.oneWait.iopTable_isSome_cls FlockDeploy.sha512 P x t.1 hx d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g'
        hgg hg)
    T hT hNs hKs hKm hBc hX hI

end SpotFlock
