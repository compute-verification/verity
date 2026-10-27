import Properties.Verity.Network
import Definitions.Core.Assumptions
import Proofs.Properties.NetworkCertifier.Period
import Proofs.Properties.CertifierDevice.Lms
import Proofs.Core.Game.Prob

/-!
# The network certifier's target: its honest and nontrivial lemmas

`Verity.Vocab.NetworkCertifierSound` (ONE-12's target, `Target .network`) bounds the probability that the network
verdict accepts (the committed blob passed the file check, NCI accepted the registration, and onsite's network
component accepts what the prover sent) while more than `K` bits went through the network certifiers, by the network
certifier's error at the strategy. Its two lemmas:

* honest (`networkComponentBound_honest`): at the honest deployed instance the compiled `ComponentAccepts .network`
  accepts what the prover sent, and where the deployment's honest prover stays within `K` (`Vocab.HonestHolds .network`)
  the target's event at it has probability 0. The instance is a period of the rows form under the w63 class's schedule
  (`NetworkBound.w63`) whose every link-window's three records and both rows' heads the certifier's device signed
  (`NetworkBound.DeviceSigned`), with `trusted-host` (`Verity.Assumptions.TrustedHost`) holding of the device with LMS's
  `signAt` under its key as its code, the key one `Lms.key` derives at parameters `Lms.Params.make` admits; whose rows
  are the scan's honest registrations (`NetworkBound.RowsSigned`: `warden_rows.check_registration`'s conditions, held by
  the scan's registration); and whose scans, at one δ, and audits the proof service accepted. LMS's completeness
  (`Properties.CertifierDevice.Proofs.Lms.lmsCorrect`) and the period check's converse
  (`Properties.NetworkCertifier.periodAccepts_of`) carry it. What stays a hypothesis is the proof service's: the
  answers' acceptance (`PeriodRowsProved`, `PeriodWindowsAt`, `PeriodAuditsProved`) and what the prover sent decoding
  to the period (`NetworkPeriod`).
* nontrivial (`networkComponentBound_nontrivial`): with the compiled component replaced by one that accepts everything
  (`Vocab.NetworkSoundBy`, which at `ComponentAccepts .network` is the target, `Vocab.networkCertifierSound_by`), a
  strategy whose plays pass the file check and NCI with probability 1 and every play of which sends more than `K` makes
  the event certain, so the target fails wherever the network certifier's error at it is below 1: where its finder
  meets its collision premise, its published `δ` plus its loss (`error_of_finderCR`).
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
    PeriodRowsProved s reg x

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
/-- **The network certifier's target's honest lemma**: at the honest deployed instance (a period of the rows form
under the w63 class's schedule, whose every link-window the certifier's device certified under its key, `trusted-host`
holding, with the scan's honest registrations, and whose scans, at one δ, and audits the proof service accepted), the
compiled `ComponentAccepts .network` accepts what the prover sent; and where the deployment's honest prover stays
within `K`, the target's event at it has probability 0, so the target's bound holds there. -/
theorem networkComponentBound_honest {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    {Dev : Type} {dev : Dev} {run : Dev → Nat × ByteArray → Except String Lms.Signature} {N W H : Nat} {toy : Bool}
    {I seed : ByteArray} {k : Lms.PrivateKey} {s : Sent} {p : NetworkCheck.Period.Params}
    {cm : NetworkCheck.Period.Commitment} {e : NetworkCheck.Period.Evidence} {sd : ByteArray}
    {scan : NetworkCheck.Period.Scan} {δ : Core.Ratio} (htrust : Assumptions.TrustedHost dev run (lmsCode k))
    (hmake : Lms.Params.make N W H toy = .ok p.lms) (hkey : Lms.key p.lms I seed = .ok k) (hpub : p.key = k.publicKey)
    (hform : p.form = .rows w63 sd scan) (hn : NetworkPeriod s p cm e) (hcw : cm.windows.size = p.windows)
    (hew : e.windows.size = p.windows)
    (hwin : ∀ (i : Nat) (reg : NetworkCheck.Period.Registration) (x : NetworkCheck.Period.Window),
      cm.windows[i]? = some reg → e.windows[i]? = some x → RowsWindowHonest dev run k p w63 sd (p.first + i) reg x)
    (hδ : Properties.NetworkCertifier.PeriodWindowsAt δ e)
    (ha : Properties.NetworkCertifier.PeriodAuditsProved p cm e) :
    ComponentAccepts .network s ∧
      (V.HonestHolds .network → ∀ θ F c : ByteArray, V.Registered θ F →
        Game.prob (fun o => V.Verdict .network o ∧ ¬ V.Holds θ F c (V.honest θ F c) .network o) (V.game θ F c)
          (V.honest θ F c) = 0) := by
  have hone : Properties.NetworkCertifier.PeriodScansOneDelta p e := by
    rw [Properties.NetworkCertifier.PeriodScansOneDelta, hform]
    exact ⟨δ, hδ⟩
  refine ⟨componentAccepts_network_of hn
    (Properties.NetworkCertifier.periodAccepts_of ⟨hcw, hew, fun i reg x hc he => ?_⟩ hone ha), ?_⟩
  · rw [hform]
    exact rowsWindowAccepted_of_honest htrust hmake hkey hpub (hwin i reg x hc he)
  · intro hH θ F c hreg
    refine nonpos_iff_eq_zero.1 (le_trans (Game.prob_mono ?_ _ _) (hH θ F c hreg).le)
    exact fun _ ho => ho.2

/-- **The network certifier's error where its finder meets its collision premise**: its published `δ` at the
strategy's SHA-512 calls plus its loss at the probability that the finder outputs a collision. -/
theorem error_of_finderCR {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α) (a : Audit)
    (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (hcr : V.FinderCR a θ F c σ) :
    V.error a θ F c σ = V.delta θ a (V.cost θ F c σ) + V.loss a θ ((V.live θ F c).collides sha512Msg a σ) := by
  simp [Vocab.error, Accounting.condError, hcr, Accounting.PeriodLive.error]

/-- **The network certifier's target's nontrivial lemma**: with the compiled component replaced by one that accepts
everything, a strategy whose plays pass the file check and NCI with probability 1, every play of which sends more than
`K` bits through the network certifiers, makes the target's event certain, so the target fails wherever the network
certifier's error at the strategy is below 1. -/
theorem networkComponentBound_nontrivial {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (acc : Sent → Prop) (hacc : ∀ s, acc s)
    (hreg : Game.prob (fun o => V.Registered o.params o.file ∧ RegistrationAccepts o.sent) (V.game θ F c) σ = 1)
    (hK : ∀ o, ¬ V.Holds θ F c σ .network o) (hlt : V.error .network θ F c σ < 1) : ¬ V.NetworkSoundBy acc := by
  intro h
  have h1 : Game.prob (fun o => (V.Registered o.params o.file ∧ RegistrationAccepts o.sent ∧ acc o.sent) ∧
      ¬ V.Holds θ F c σ .network o) (V.game θ F c) σ = 1 :=
    le_antisymm (Game.prob_le_one _ _ σ)
      (hreg.symm.le.trans (Game.prob_mono (fun o ho => ⟨⟨ho.1, ho.2, hacc _⟩, hK o⟩) _ σ))
  exact absurd (h1.symm.le.trans (h θ F c σ)) (not_le.2 hlt)

end Verity
