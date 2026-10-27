import Properties.Verity.Network
import Proofs.Properties.NetworkCertifier.Period
import Proofs.Properties.CertifierDevice.Lms
import Proofs.Core.Game.Prob

/-!
# The network certifier's component bound: its honest and nontrivial lemmas

`Verity.Vocab.ComponentBound .network` (ONE-12's target at the component, `Properties.Verity.Legs`) bounds the
probability that onsite's network component accepts what the prover sent, every release its check read is in the
device's log, more than `K` bits went through the network certifiers, and there is no break. Its two lemmas:

* honest (`networkComponentBound_honest`): at the honest deployed instance the compiled `ComponentAccepts .network`
  accepts what the prover sent, and wherever the play stays within `K` the bound's event has probability 0. The
  instance is a period of the rows form under the w63 class's schedule (`NetworkBound.w63`) whose every link-window's
  three records and both rows' heads the certifier's device signed (`NetworkBound.DeviceSigned`), with `trusted-host`
  (`Verity.Assumptions.TrustedHost`) holding of the device with LMS's `signAt` under its key as its code, the key one
  `Lms.key` derives at parameters `Lms.Params.make` admits; whose rows are the scan's honest registrations
  (`NetworkBound.RowsSigned`: `warden_rows.check_registration`'s conditions, held by the scan's registration); and
  whose scans and audits the proof service accepted. LMS's completeness
  (`Properties.CertifierDevice.Proofs.Lms.lmsCorrect`) and the period check's converse
  (`Properties.NetworkCertifier.periodAccepts_of`) carry it. What stays a hypothesis is the proof service's: the answers' acceptance (`PeriodRowsProved`, `PeriodAuditsProved`) and what the prover sent
  decoding to the period (`NetworkPeriod`).
* nontrivial (`networkComponentBound_nontrivial`): with the compiled check replaced by one that accepts everything
  (`Vocab.NetworkBoundBy`, which at `ComponentAccepts .network` is the bound, `Vocab.componentBound_network_by`), a
  strategy whose every play sends more than `K`, with every release read logged and no break, makes the event certain,
  so the bound fails wherever the published error is below 1. `NetworkBound.beyondK` is such an instance, closed
  (`networkBoundBy_beyondK_fails`).
-/

namespace Verity

open scoped ENNReal

namespace NetworkBound

open NetworkCheck.Period Properties.NetworkCertifier

local instance instLawfulBEqByteArray_networkBound : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-- The w63 class's schedule (`benchmarks/network_traces/scan_zk.py`'s `PARAMS`): 100 ms buckets of 512-byte cells, 4
lanes of 4 slots, a delay of 21, 32-bucket link-windows. -/
def w63 : NetworkCheck.Period.Schedule :=
  { bucketMs := 100, cellBytes := 512, delay := 21, egressCells := 65, ingressBuckets := 12, ingressCells := 47,
    lanes := 4, queue := 4, slots := 4, window := 32 }

/-- The certifier's device's code: LMS's signature under its key `k` at leaf `q` of the message `m`. -/
def lmsCode (k : Lms.PrivateKey) (qm : Nat × ByteArray) : Except String Lms.Signature := Lms.signAt k qm.1 qm.2

/-- **The certifier's device signed certificate `x`** of `link`'s end `dir` at window `w` under its key `k`: `x` names
that end and window and the key (its `I`), and its signature is what the device ran (`run dev`) at some leaf on
`certificate_message` of its fields in the session. -/
def DeviceSigned {Dev : Type} (dev : Dev) (run : Dev → Nat × ByteArray → Except String Lms.Signature)
    (k : Lms.PrivateKey) (p : NetworkCheck.Period.Params) (link : String) (dir : NetworkCheck.Direction) (w : Nat)
    (x : NetworkCheck.Check.Certificate) : Prop :=
  x.direction = dir ∧ x.link = link ∧ x.window = w ∧ x.keyId = k.I ∧
    ∃ m q sig, NetworkCheck.certificateMessage x.keyId p.session x.link x.direction x.window x.digest = .ok m ∧
      run dev (q, m) = .ok sig ∧ x.signature = sig.encode

/-- **Link-window `w`'s rows in direction `dir`, honestly registered**: `PeriodRowsChecked`'s conditions on the rows
under the schedule `s` whose digest is `sd`, with the certificate on their head the device's (`DeviceSigned`). -/
def RowsSigned {Dev : Type} (dev : Dev) (run : Dev → Nat × ByteArray → Except String Lms.Signature)
    (k : Lms.PrivateKey) (p : NetworkCheck.Period.Params) (s : NetworkCheck.Period.Schedule) (sd : ByteArray) (w : Nat)
    (d : NetworkCheck.Direction) (r : NetworkCheck.Period.Rows) : Prop :=
  (r.format == rowsFormat && r.value == s!"{p.link}/{d.value}/{w}" && r.link == p.link && r.direction == d.value &&
      r.window == w) = true ∧
    (r.parameters == Flock.hexOf sd) = true ∧
    (r.cellBytes == s.cellBytes && r.rows == s.window && r.words == s.allocation * s.cellBytes / 2 &&
      r.wordBits == outBits && r.rowSchema == rowSchema &&
      r.layout == Layout.mk s.window s.allocation s.slots (w * s.window)) = true ∧
    ∃ dom head, rowsDomain s sd p.session p.link d w = .ok dom ∧ (r.domain == Flock.hexOf dom) = true ∧
      headDigest p.link d w sd s r.wiring dom r.root = .ok head ∧ (r.digest == head) = true ∧
      DeviceSigned dev run k p p.link d w
        { keyId := r.key, link := p.link, direction := d, window := w, digest := r.digest, signature := r.signature }

/-- **An honest rows-form link-window `w`** (registered as `reg`, answered by `x`): its registration names the scan's
Program, its three records (the link's ingress and egress, `<link>-fails`' egress) the device signed, its ingress then
egress rows honestly registered and held by the scan's registration, and its scan proved. -/
def RowsWindowHonest {Dev : Type} (dev : Dev) (run : Dev → Nat × ByteArray → Except String Lms.Signature)
    (k : Lms.PrivateKey) (p : NetworkCheck.Period.Params) (s : NetworkCheck.Period.Schedule) (sd : ByteArray) (w : Nat)
    (reg : Registration) (x : NetworkCheck.Period.Window) : Prop :=
  PeriodWindowProgram p reg ∧
    (∃ rq rp fl, x.certificates = #[rq, rp, fl] ∧ DeviceSigned dev run k p p.link .ingress w rq ∧
      DeviceSigned dev run k p p.link .egress w rp ∧ DeviceSigned dev run k p (p.link ++ "-fails") .egress w fl) ∧
    (∃ ri re, x.shown = .rows #[ri, re] ∧ RowsSigned dev run k p s sd w .ingress ri ∧
      RowsSigned dev run k p s sd w .egress re ∧ holds reg ri = true ∧ holds reg re = true) ∧
    PeriodRowsProved reg x

theorem signs_of {p : NetworkCheck.Period.Params} {x : NetworkCheck.Check.Certificate} {m : ByteArray}
    {sig : Lms.Signature} (hk : x.keyId = p.key.extract 0 16)
    (hm : NetworkCheck.certificateMessage x.keyId p.session x.link x.direction x.window x.digest = .ok m)
    (hs : Lms.Signature.decode p.lms x.signature = .ok sig) (hv : Lms.verify p.lms p.key m sig = .ok true) :
    signs p x = true := by
  unfold signs
  rw [hm, hs]
  simp [hk, hv, Except.toOption]

variable {Dev : Type} {dev : Dev} {run : Dev → Nat × ByteArray → Except String Lms.Signature} {N W H : Nat}
  {toy : Bool} {I seed : ByteArray} {k : Lms.PrivateKey} {p : NetworkCheck.Period.Params}

/-- **A certificate the device signed is certified**, by LMS's completeness. -/
theorem certifiesOn_of_signed {link : String} {dir : NetworkCheck.Direction} {w : Nat}
    {x : NetworkCheck.Check.Certificate} (htrust : Assumptions.TrustedHost dev run (lmsCode k))
    (hmake : Lms.Params.make N W H toy = .ok p.lms) (hkey : Lms.key p.lms I seed = .ok k) (hpub : p.key = k.publicKey)
    (h : DeviceSigned dev run k p link dir w x) : certifiesOn p link dir w x = true := by
  obtain ⟨hd, hl, hw, hxk, m, q, sig, hm, hrun, hsx⟩ := h
  obtain ⟨-, hI, -, hIs, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hkey
  have hid : p.key.extract 0 16 = k.I := by
    rw [hpub, Lms.PrivateKey.publicKey]
    exact ByteArray.extract_append_eq_left (hI ▸ hIs).symm
  have hsig : Lms.signAt k q m = .ok sig := (htrust (q, m)).symm.trans hrun
  obtain ⟨hv, hdec⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsCorrect N W H toy p.lms I seed k q m sig hmake hkey hsig
  have hs : signs p x = true := signs_of (m := m) (sig := sig) (hxk.trans hid.symm) hm (hsx ▸ hdec) (hpub ▸ hv)
  simp [certifiesOn, hd, hl, hw, hs]

theorem rowsChecked_of_signed {s : NetworkCheck.Period.Schedule} {sd : ByteArray} {w : Nat}
    {d : NetworkCheck.Direction} {r : NetworkCheck.Period.Rows} (htrust : Assumptions.TrustedHost dev run (lmsCode k))
    (hmake : Lms.Params.make N W H toy = .ok p.lms) (hkey : Lms.key p.lms I seed = .ok k) (hpub : p.key = k.publicKey)
    (h : RowsSigned dev run k p s sd w d r) : PeriodRowsChecked p s sd w d r := by
  obtain ⟨c1, c2, c3, dom, head, hdom, c4, hhead, c5, hsig⟩ := h
  exact ⟨c1, c2, c3, dom, head, hdom, c4, hhead, c5, certifiesOn_of_signed htrust hmake hkey hpub hsig⟩

/-- **An honest rows-form link-window is accepted in the parameters' form.** -/
theorem rowsWindowAccepted_of_honest {s : NetworkCheck.Period.Schedule} {sd : ByteArray} {w : Nat}
    {reg : Registration} {x : NetworkCheck.Period.Window} (htrust : Assumptions.TrustedHost dev run (lmsCode k))
    (hmake : Lms.Params.make N W H toy = .ok p.lms) (hkey : Lms.key p.lms I seed = .ok k) (hpub : p.key = k.publicKey)
    (h : RowsWindowHonest dev run k p s sd w reg x) : PeriodRowsWindowAccepted p s sd w reg x := by
  obtain ⟨hp, ⟨rq, rp, fl, hcs, h1, h2, h3⟩, ⟨ri, re, hx, hi, he, hhi, hhe⟩, hpr⟩ := h
  exact ⟨hp, ⟨rq, rp, fl, hcs, certifiesOn_of_signed htrust hmake hkey hpub h1,
      certifiesOn_of_signed htrust hmake hkey hpub h2, certifiesOn_of_signed htrust hmake hkey hpub h3⟩,
    ⟨ri, re, hx, rowsChecked_of_signed htrust hmake hkey hpub hi, rowsChecked_of_signed htrust hmake hkey hpub he, hhi,
      hhe⟩, hpr⟩

end NetworkBound

open NetworkBound in
/-- **`ComponentBound .network`'s honest lemma**: at the honest deployed instance (a period of the rows form under the
w63 class's schedule, whose every link-window the certifier's device certified under its key, `trusted-host` holding,
with the scan's honest registrations, and whose scans and audits the proof service accepted), the compiled
`ComponentAccepts .network` accepts what the prover sent, and in every play that stays within `K` the bound's event
has probability 0. -/
theorem networkComponentBound_honest {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    {Dev : Type} {dev : Dev} {run : Dev → Nat × ByteArray → Except String Lms.Signature} {N W H : Nat} {toy : Bool}
    {I seed : ByteArray} {k : Lms.PrivateKey} {s : Sent} {p : NetworkCheck.Period.Params}
    {cm : NetworkCheck.Period.Commitment} {e : NetworkCheck.Period.Evidence} {sd : ByteArray}
    {scan : NetworkCheck.Period.Audit} (htrust : Assumptions.TrustedHost dev run (lmsCode k))
    (hmake : Lms.Params.make N W H toy = .ok p.lms) (hkey : Lms.key p.lms I seed = .ok k) (hpub : p.key = k.publicKey)
    (hform : p.form = .rows w63 sd scan) (hn : NetworkPeriod s p cm e) (hcw : cm.windows.size = p.windows)
    (hew : e.windows.size = p.windows)
    (hwin : ∀ (i : Nat) (reg : NetworkCheck.Period.Registration) (x : NetworkCheck.Period.Window),
      cm.windows[i]? = some reg → e.windows[i]? = some x → RowsWindowHonest dev run k p w63 sd (p.first + i) reg x)
    (ha : Properties.NetworkCertifier.PeriodAuditsProved p cm e) :
    ComponentAccepts .network s ∧
      ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)), (∀ o, V.Holds θ F c σ .network o) →
        Game.prob (fun o => ComponentAccepts .network o.sent ∧ (∀ r ∈ V.coinsRead .network o.sent, r ∈ o.log) ∧
          ¬ V.Holds θ F c σ .network o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ = 0 := by
  refine ⟨componentAccepts_network_of hn
    (Properties.NetworkCertifier.periodAccepts_of ⟨hcw, hew, fun i reg x hc he => ?_⟩ ha), ?_⟩
  · rw [hform]
    exact rowsWindowAccepted_of_honest htrust hmake hkey hpub (hwin i reg x hc he)
  · intro θ F c σ hK
    refine nonpos_iff_eq_zero.1 (le_trans (Game.prob_mono ?_ _ σ) (Game.prob_false _ σ).le)
    exact fun o h => (h.2.2.1 (hK o)).elim

/-- **`ComponentBound .network`'s nontrivial lemma**: with the compiled check replaced by one that accepts everything,
a strategy whose every play sends more than `K` bits through the network certifiers, with every release read logged
and no break, makes the bound's event certain, so the bound fails wherever the published error is below 1. -/
theorem networkComponentBound_nontrivial {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (acc : Sent → Prop) (hacc : ∀ s, acc s)
    (hlog : ∀ o : Outcome, ∀ r ∈ V.coinsRead .network o.sent, r ∈ o.log)
    (hK : ∀ o, ¬ V.Holds θ F c σ .network o) (hB : ∀ o, ¬ V.Break θ F c o) (hδ : V.delta θ .network < 1) :
    ¬ V.NetworkBoundBy acc := by
  intro h
  have h1 : Game.prob (fun o => acc o.sent ∧ (∀ r ∈ V.coinsRead .network o.sent, r ∈ o.log) ∧
      ¬ V.Holds θ F c σ .network o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ = 1 :=
    le_antisymm (Game.prob_le_one _ _ σ)
      ((Game.prob_true _ σ).symm.le.trans (Game.prob_mono (fun o _ => ⟨hacc _, hlog o, hK o, hB o⟩) _ σ))
  exact absurd (h1.symm.le.trans (h θ F c σ)) (not_le.2 hδ)

namespace NetworkBound

/-- The empty circuit: the network certifier's bound doesn't read the circuit. -/
def emptyCircuit : Verity.Circuit where
  N := 0
  w := fun g => g.elim0
  isInput := fun g => g.elim0
  reads := fun g => g.elim0
  reads_lt := fun g => g.elim0
  input_reads := fun g => g.elim0
  op := fun g => g.elim0
  op_congr := fun g => g.elim0
  outputs := ∅

/-- A device with no resources and no side channel. -/
def none0 : Accounting.Device Unit := ⟨0, 0, fun _ => 0, 0, fun _ => 0⟩

/-- A tick in which nothing runs. -/
def idle : Accounting.Step Unit Unit := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩

theorem idle_fits : idle.Fits none0 := by
  unfold Accounting.Step.Fits
  refine ⟨fun o => ?_, ?_, fun _ => ?_, ?_, ?_⟩ <;> simp [idle, none0]

/-- **A vocabulary whose every play sends more than `K`**: no rounds and no releases read, the registered file the
empty one, and a world in which the network certifiers saw more than any `K` (`commWithin` false), at a published
error of 0. -/
noncomputable def beyondK : Vocab none0 Unit where
  circuitOf := fun _ => emptyCircuit
  units := fun _ => 0
  partsOf := fun _ => ⟨fun g => g.elim0, fun g => g.elim0⟩
  place := fun _ _ _ _ => none
  programDigest := fun _ => none
  registered := fun _ => .empty
  rounds := fun _ => []
  coinsRead := fun _ _ => []
  world := fun _ _ _ _ _ =>
    { acct := fun _ => (), win := ⟨0, 0⟩, schedule := fun _ => idle, fits := fun _ => idle_fits,
      commWithin := fun _ => False }
  certOf := fun _ => ⟨0, 0, 0, 0, 0⟩
  priceOf := fun _ _ => 0
  delta := fun _ _ => 0

theorem beyondK_noBreak (o : Outcome) : ¬ beyondK.Break .empty .empty .empty o := by
  rintro (⟨_, -, _, hr, -⟩ | ⟨g, -⟩ | ⟨hF, -⟩)
  · exact absurd hr List.not_mem_nil
  · exact g.elim0
  · exact hF rfl

/-- **The network certifier's bound fails for a check that accepts everything**, at `beyondK`: the empty file and
commitment, the strategy that sends empty evidence. -/
theorem networkBoundBy_beyondK_fails (acc : Sent → Prop) (hacc : ∀ s, acc s) : ¬ beyondK.NetworkBoundBy acc :=
  networkComponentBound_nontrivial beyondK .empty .empty .empty ⟨.empty, PUnit.unit⟩ acc hacc
    (fun _ _ hr => absurd hr List.not_mem_nil) (fun _ h => h) beyondK_noBreak (by simp [beyondK])

end NetworkBound

end Verity
