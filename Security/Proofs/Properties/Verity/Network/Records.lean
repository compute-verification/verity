import Proofs.Properties.Verity.Subprotocols
import Proofs.Properties.Verity.Network.Window
import Proofs.Properties.Verity.Network.Legs
import Proofs.Properties.CertifierNetwork.Runs

/-!
# S8's record premises at the period's certificates: the network certifier's record off the forger's event

`Verity.Network.windowLeOff` reads two premises of the network certifier's record beside the named steps: the record
covers the horizon off an event (`RecordCoversOff`), and the trace holds the registered values
(`WindowValuesRegistered`). Here both come from what the period check reads in the clear and what the device's code
does, up to one event whose probability the bound gains (`CertBreak`) and no lemma here bounds (the break, below).

* **The certificates** (`periodCerts`): the period check accepted `x` as window `i`'s certificate of boundary link `c`
  where it `certifies` `x` for the link's end `dir c` at the window `first + i`; the device certified the execution
  phase's transfer blocks `x` names where its output holds a certificate naming the same link, end and window. Where
  onsite's network component accepts, every window's every boundary link has an accepted certificate, in both forms
  (`certifiedOnAccept_period`, from S1: the clock form's egress and ingress certificates, the rows form's requests'
  and responses').
* **The device** (`LogIssued`): each certificate of its output is one `Issuer.certify` returned for a record and an
  end, and a complete record of the transfer blocks the period names as window `i`'s boundary link `c` is a complete
  status of `c` in the record format's window `i`: `trusted-host`, and the record format faithful to the device's
  record. The issuer certifies only a complete record (`certify_complete`), so a certificate the device issued for an
  accepted certificate's transfer blocks gives that status (`SignedComplete`, `signed_complete`).
* **The break** (`certBreak_of_forged`): where the period accepted a certificate of transfer blocks the device never
  certified (`CertForged`), that certificate verifies under the period's key over its message in the period's session,
  and either the device never signed that message in its session (`Signed`, `euf-cma/lms-sha512-n64`) or the message
  is one the device signed for another certificate, whose message input differs from this one's under one SHA-512
  digest (`ecr/sha-512`). The period's parameters are the auditor's, naming the device's key and session. `Signed`
  ranges over the device's record certificates alone, so the first case is an LMS forgery only against a device that
  signs nothing else in the session. A device that also signs a clock certificate (`Issuer.salted_clock`) or a rows head
  (`warden_rows.attest`) of the same link, end and window, whatever the record's status, doesn't meet that: the
  period's `certifies` reads neither digest, so it accepts either in the record certificate's place. In the clock form
  the window's proof opens both certificates' signed digests to complete heads (N2's head check, one per end, at
  `NetworkCheck.LinkWindow.pubRow`), which this route, in the clear, doesn't read; the rows form runs no verifier on
  the window.

The clock form's record then covers the horizon off `CertBreak` (`recordCoversOff_break_clock`), and the rows form's
where the format's jitter class is below `Dj` (`RecordsRows`, `recordCoversOff_break_rows`). A window whose registration
`sw i` registers no values, as the clock form's execution phase, closes the ρ gap where its statement is registered as
`sw i` at every coin (`windowValuesRegistered_of_sw`); one that registers values leaves it open. The bound gains the
break's probability (`windowLe_clock`, `windowLe_rows`), or reads none where S5's guard excludes the break
(`windowLe_off`, and the refinement at a verdict, `networkWindowLe_off`). At a coupling whose break holds `CertBreak`,
the verdict's break being absent is that guard (`couplingGuard`), so the refinement at its verdict reads S8's coverage
as no premise (`networkWindowLe_verdict_clock`, `networkWindowLe_verdict_rows`). The ρ gap stays one in both forms: the
coupling's statement is registered as `sw i` only under the guard, and the ρ gap is read at every coin.
-/

namespace Verity.Network

open Accounting Accounting.JointBridge Claims Claims.Sampling NetTiming Properties.NetworkCertifier

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
  [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ}
  {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T}
  {reg : AuditStatement → Finset ℕ} {Dj n : ℕ}
  {r : PeriodRun d α C Ω} {K : ℝ}

variable {R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n}

/-- What the period's `certifiesOn` checks of a certificate: its end, link and window, and an LMS signature that decodes
and verifies under the parameters' key over the certificate's message in their session. -/
theorem certifiesOn_spec {p : NetworkCheck.Period.Params} {link : String} {dr : NetworkCheck.Direction} {w : ℕ}
    {x : NetworkCheck.Check.Certificate} (h : NetworkCheck.Period.certifiesOn p link dr w x = true) :
    x.direction = dr ∧ x.link = link ∧ x.window = w ∧ ∃ m sig,
      NetworkCheck.certificateMessage x.keyId p.session x.link x.direction x.window x.digest = .ok m ∧
      Lms.Signature.decode p.lms x.signature = .ok sig ∧ Lms.verify p.lms p.key m sig = .ok true := by
  unfold NetworkCheck.Period.certifiesOn NetworkCheck.Check.Certificate.signs at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨hd, hl⟩, hw⟩, -, hs⟩ := h
  refine ⟨hd, hl, hw, ?_⟩
  split at hs
  · rename_i m sig hm hsig
    refine ⟨m, sig, hm, hsig, ?_⟩
    cases hv : Lms.verify p.lms p.key m sig with
    | error e => rw [hv] at hs; cases hs
    | ok b => rw [hv] at hs; cases b <;> first | rfl | cases hs
  · cases hs

/-! ## The certificates -/

/-- **The period's certificates and the device's** at the coins `ξ`: the period onsite's network case decodes to
accepted `x` as window `i`'s certificate of boundary link `c` where `x` is one of the window's certificates and the
period `certifies` it as the link's end `dir c` at the window `first + i`; the device certified the transfer blocks `x`
names where a certificate of its output `out ξ` names the same link, end and window. -/
def periodCerts (s : ((i : I) → Ω i) → Sent) (dir : Fin Kₒ → NetworkCheck.Direction)
    (out : ((i : I) → Ω i) → DeviceOutput) : CertRuns R NetworkCheck.Check.Certificate where
  accepted ξ i c x := ∃ p cm e win, NetworkPeriod .onsite (s ξ) p cm e ∧ e.windows[i.val]? = some win ∧
    x ∈ win.certificates ∧ NetworkCheck.Period.certifies p (dir c) (p.first + i.val) x = true
  signed ξ x := ∃ cr ∈ (out ξ).certificates, cr.certificate.link = x.link ∧
    cr.certificate.direction = x.direction ∧ cr.certificate.window = x.window

/-- **The period checks each window's certificates in the clear** (S1): where onsite's network component accepts and
its period has the realization's `n` windows at least (S6's `windows`), every boundary link of every window has a
certificate the period accepted, of either end: the clock form's egress or ingress certificate, the rows form's
responses' or requests'. -/
theorem certifiedOnAccept_period {s : ((i : I) → Ω i) → Sent} (dir : Fin Kₒ → NetworkCheck.Direction)
    (out : ((i : I) → Ω i) → DeviceOutput) (hn : ∀ ξ p c e, NetworkPeriod .onsite (s ξ) p c e → n ≤ p.windows) :
    CertifiedOnAccept s (periodCerts (R := R) s dir out) := by
  intro ξ i c hca
  obtain ⟨p, cm, e, hnp, hc, he, hall⟩ := componentAccepts_network_windows periodAcceptMeans hca
  have hi : i.val < p.windows := lt_of_lt_of_le i.isLt (hn ξ p cm e hnp)
  have hc' : i.val < cm.windows.size := by omega
  have he' : i.val < e.windows.size := by omega
  have hw := hall i.val cm.windows[i.val] e.windows[i.val] (Array.getElem?_eq_getElem hc')
    (Array.getElem?_eq_getElem he')
  have key : ∀ x, x ∈ e.windows[i.val].certificates →
      NetworkCheck.Period.certifies p (dir c) (p.first + i.val) x = true →
      ∃ x, (periodCerts (R := R) s dir out).accepted ξ i c x :=
    fun x hx hcert => ⟨x, p, cm, e, e.windows[i.val], hnp, Array.getElem?_eq_getElem he', hx, hcert⟩
  split at hw
  · obtain ⟨-, ⟨eg, ig, hcs, heg, hig⟩, -, -⟩ := hw
    cases hd : dir c with
    | egress => exact key eg (by rw [hcs]; simp) (by rw [hd]; exact heg)
    | ingress => exact key ig (by rw [hcs]; simp) (by rw [hd]; exact hig)
  · obtain ⟨-, ⟨rq, rp, fl, hcs, hrq, hrp, -⟩, -, -⟩ := hw
    cases hd : dir c with
    | egress => exact key rp (by rw [hcs]; simp) (by rw [hd]; exact hrp)
    | ingress => exact key rq (by rw [hcs]; simp) (by rw [hd]; exact hrq)

/-! ## The device -/

/-- **The device certifies with its issuer, of the records the format reads (`trusted-host`, and the record format
faithful to the device's record)**: each certificate of the device's output `out ξ` is one `Issuer.certify` returned
for a record and an end, and where the period onsite's network case decodes to names that record's transfer blocks as
window `i`'s boundary link `c` (the period's link, `c`'s end `dir c` and window `first + i`) and the record is complete,
`P ξ i c` holds: in the record format, `c`'s status in window `i` is complete. -/
def LogIssued (s : ((i : I) → Ω i) → Sent) (dir : Fin Kₒ → NetworkCheck.Direction)
    (out : ((i : I) → Ω i) → DeviceOutput) (P : ((i : I) → Ω i) → Fin n → Fin Kₒ → Prop) : Prop :=
  ∀ ξ, ∀ cr ∈ (out ξ).certificates, ∃ (iss iss' : CertifierDevice.Issuer) (salt : ByteArray)
      (leaf : Except String ℕ) (rec : CertifierDevice.BlockGrid) (dr : NetworkCheck.Direction) (g : NetworkCheck.Params)
      (pd : ByteArray), iss.certify salt leaf rec dr g pd = .ok (iss', some cr) ∧
    ∀ (i : Fin n) c p cm e, NetworkPeriod .onsite (s ξ) p cm e → rec.link = p.link → dr = dir c →
      rec.window = p.first + i.val → rec.status = .complete → P ξ i c

/-- **B at the period's certificates**: where the period accepted `x` as window `i`'s certificate of boundary link `c`
and the device certified the transfer blocks `x` names, the device's issuer certified a record of them, which is
complete (`certify_complete`), so `P ξ i c`. -/
theorem signed_complete {s : ((i : I) → Ω i) → Sent} {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : I) → Ω i) → DeviceOutput} {P : ((i : I) → Ω i) → Fin n → Fin Kₒ → Prop}
    (hlog : LogIssued s dir out P) (ξ : (i : I) → Ω i) (i : Fin n) (c : Fin Kₒ) (x : NetworkCheck.Check.Certificate)
    (ha : (periodCerts (R := R) s dir out).accepted ξ i c x) (hs : (periodCerts (R := R) s dir out).signed ξ x) :
    P ξ i c := by
  obtain ⟨p, cm, e, win, hnp, -, -, hcert⟩ := ha
  obtain ⟨cr, hcr, hl, hd, hw⟩ := hs
  obtain ⟨hxd, hxl, hxw, -⟩ := certifiesOn_spec (link := p.link) hcert
  obtain ⟨iss, iss', salt, leaf, rec, dr, g, pd, hiss, hP⟩ := hlog ξ cr hcr
  obtain ⟨-, -, hcl, hcd, hcw⟩ := certify_some hiss
  exact hP i c p cm e hnp (by rw [← hcl, hl, hxl]) (by rw [← hcd, hd, hxd]) (by rw [← hcw, hw, hxw])
    (Properties.CertifierDevice.Proofs.Issuer.certify_complete hiss)

/-- **B at the clock form** (`SignedComplete`): `signed_complete` where `P` is the clock form's status. -/
theorem signedComplete_period (W : WindowRuns R) (F : ClockFormat R.Conn R.Cell L Kₒ S)
    {s : ((i : I) → Ω i) → Sent} {dir : Fin Kₒ → NetworkCheck.Direction} {out : ((i : I) → Ω i) → DeviceOutput}
    (hlog : LogIssued s dir out fun ξ (i : Fin n) c => F.status (W.sched ξ) (R.arr (W.srv ξ)) i.val c = .complete) :
    SignedComplete W F (periodCerts (R := R) s dir out) :=
  signed_complete hlog

/-! ## The break -/

/-- **The break the forger's event carries at `ξ`**: in the period onsite's network case decodes to, a certificate `x`
of transfer blocks verifies under the period's key over its message `m` in the period's session, and either the device
never signed `m` in its session `sess ξ` as a record certificate (`Signed`: an LMS forgery, `euf-cma/lms-sha512-n64`,
where the device signs nothing else in the session), or one of the device's certificates has a message input other
than `x`'s with one SHA-512 digest (`ecr/sha-512`). No lemma here bounds its probability. -/
def CertBreak (s : ((i : I) → Ω i) → Sent) (sess : ((i : I) → Ω i) → ByteArray)
    (out : ((i : I) → Ω i) → DeviceOutput) (ξ : (i : I) → Ω i) : Prop :=
  ∃ (p : NetworkCheck.Period.Params) (cm : NetworkCheck.Period.Commitment) (e : NetworkCheck.Period.Evidence)
    (j : ℕ) (win : NetworkCheck.Period.Window) (x : NetworkCheck.Check.Certificate) (m : ByteArray),
    NetworkPeriod .onsite (s ξ) p cm e ∧ e.windows[j]? = some win ∧ x ∈ win.certificates ∧
    NetworkCheck.certificateMessage x.keyId p.session x.link x.direction x.window x.digest = .ok m ∧
    (∃ sig, Lms.Signature.decode p.lms x.signature = .ok sig ∧ Lms.verify p.lms p.key m sig = .ok true) ∧
    (¬ Signed (sess ξ) (out ξ) m ∨
      ∃ cr ∈ (out ξ).certificates, ∃ u v,
        NetworkCheck.messageInput cr.certificate.keyId (sess ξ) cr.certificate.link cr.certificate.direction
          cr.certificate.window cr.certificate.digest = .ok u ∧
        NetworkCheck.messageInput x.keyId p.session x.link x.direction x.window x.digest = .ok v ∧
        u ≠ v ∧ Flock.Sha512.hash u = Flock.Sha512.hash v)

/-- **The forger's event is `CertBreak`.** Where the period accepted a certificate of transfer blocks the
device never certified, its message is one the device never signed, or the device signed it over another message
input: one of equal input names the same transfer blocks (`messageInput_inj`, the device's link being an encodable
head's, `link_short`). -/
theorem certBreak_of_forged {s : ((i : I) → Ω i) → Sent} {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : I) → Ω i) → DeviceOutput} {P : ((i : I) → Ω i) → Fin n → Fin Kₒ → Prop}
    (sess : ((i : I) → Ω i) → ByteArray) (hlog : LogIssued s dir out P) {ξ : (i : I) → Ω i}
    (h : CertForged (periodCerts (R := R) s dir out) ξ) : CertBreak s sess out ξ := by
  obtain ⟨i, c, x, ⟨p, cm, e, win, hnp, hwin, hx, hcert⟩, hns⟩ := h
  obtain ⟨-, -, -, m, sig, hm, hsig, hv⟩ := certifiesOn_spec (link := p.link) hcert
  refine ⟨p, cm, e, i.val, win, x, m, hnp, hwin, hx, hm, ⟨sig, hsig, hv⟩, ?_⟩
  by_cases hS : Signed (sess ξ) (out ξ) m
  · right
    obtain ⟨cr, hcr, hcm⟩ := hS
    have hcm' : NetworkCheck.certificateMessage cr.certificate.keyId (sess ξ) cr.certificate.link
        cr.certificate.direction cr.certificate.window cr.certificate.digest = .ok m :=
      Properties.NetworkCertifier.toOption_eq_some.mp hcm
    obtain ⟨u, hu, rfl⟩ := certificateMessage_ok hcm'
    obtain ⟨v, hv', huv⟩ := certificateMessage_ok hm
    refine ⟨cr, hcr, u, v, hu, hv', fun he => hns ?_, huv⟩
    subst he
    obtain ⟨iss, iss', salt, leaf, rec, dr, g, pd, hiss, -⟩ := hlog ξ cr hcr
    obtain ⟨hhead, ⟨dg, hdg, -⟩, hcl, -, -⟩ := certify_some hiss
    have hb : cr.certificate.link.toUTF8.size < 2 ^ 16 := by
      rw [hcl, ← (Properties.CertifierDevice.Proofs.Issuer.recordHead_fields hhead).1]
      exact link_short hdg
    obtain ⟨-, -, hl, hd, hw, -⟩ := Properties.NetworkCertifier.Bytes.messageInput_inj hu hv' hb
    exact ⟨cr, hcr, hl, hd, by exact_mod_cast hw⟩
  · exact Or.inl hS

/-- S8's coverage off an event holds off any larger one. -/
theorem RecordCoversOff.mono {W : WindowRuns R} {s : ((i : I) → Ω i) → Sent} {B B' : ((i : I) → Ω i) → Prop}
    (h : RecordCoversOff W s B) (hB : ∀ ξ, B ξ → B' ξ) : RecordCoversOff W s B' :=
  fun ξ i hca hnb => h ξ i hca fun hb => hnb (hB ξ hb)

/-- S8's registration premise under a guard holds under any stronger one. -/
theorem WindowRegistersAs.mono {W : WindowRuns R} {s : ((i : I) → Ω i) → Sent} {G G' : ((i : I) → Ω i) → Prop}
    (h : WindowRegistersAs W s G) (hG : ∀ ξ, G' ξ → G ξ) : WindowRegistersAs W s G' :=
  fun ξ i p c e rg st hg => h ξ i p c e rg st (hG ξ hg)

/-! ## The bound off the break -/

/-- **`windowLeOff` at a guard that excludes its event**: where S5's guard `G` excludes `Brk`, the network audit's
acceptance under `G` with window `i` not good is at most `PrBad`, with no term for `Brk`: `windowLe`'s bound, S8's
coverage read off `Brk` (`RecordCoversOff`). `windowLeOff` is the unexcluded form, plus `Brk`'s probability. -/
theorem windowLe_off (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} {G Brk : ((i : I) → Ω i) → Prop}
    (h2 : PeriodRunsV2 W s) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G) (h6 : NetworkCouples R s W)
    (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s G) (hcov : RecordCoversOff W s Brk)
    (hG : ∀ ξ, G ξ → ¬ Brk ξ) :
    ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      ComponentAccepts .onsite .network (s (Function.update ξ (r.own .network) x)) ∧
        G (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt (Function.update ξ (r.own .network) x) i)
          (R.wit (Function.update ξ (r.own .network) x) i) (R.obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V (Rlt R) (E D decode) (R.stmt ξ i) (R.prover ξ i) := by
  intro ξ i
  refine pr_le_of_coins (W.coins ξ i) (h4 ξ i) fun x ⟨hca, hg, hbad⟩ c =>
    ⟨h5 ξ i x hg (h2 _ i hca), fun hx => ?_⟩
  have hex : ∃ w, Rlt R (R.stmt (Function.update ξ (r.own .network) x) i) w := by
    rw [h6.stmt]
    exact exists_of_ext hx
  exact hbad (windowGood_of_covers W h2 h6 hρ hreg hca hg (hcov _ i hca (hG _ hg)) (h6.wit _ i hex))

/-- **The network certifier's refinement off the break** (`Verity.Subprotocol.NetworkWindowLe`): `windowLe_off` at any
verdict `A` under which onsite's network component accepts and S5's guard holds, the guard excluding `Brk`. -/
theorem networkWindowLe_off {A : Audit → ((i : I) → Ω i) → Prop} {s : ((i : I) → Ω i) → Sent}
    {G Brk : ((i : I) → Ω i) → Prop} (hA : ∀ ξ, A .network ξ → ComponentAccepts .onsite .network (s ξ) ∧ G ξ)
    (W : WindowRuns R) (h2 : PeriodRunsV2 W s) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G)
    (h6 : NetworkCouples R s W) (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s G)
    (hcov : RecordCoversOff W s Brk) (hG : ∀ ξ, G ξ → ¬ Brk ξ) :
    Subprotocol.NetworkWindowLe R A := fun ξ i =>
  le_trans (Pouw.pr_mono fun _ hx => ⟨(hA _ hx.1).1, (hA _ hx.1).2, hx.2⟩)
    (windowLe_off W h2 h4 h5 h6 hρ hreg hcov hG ξ i)

/-- **The refinement in the game off the break**: the network certifier's refinement at a coupling's verdict at the
code, S8's coverage read off an event `Brk` that the coupling's break holds, so the verdict's break being absent
excludes it. -/
theorem networkWindowLe_verdict_off {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (q : V.Coupling .onsite .network θ F c σ) {K : ℝ}
    {R : RealizesCommOfPeriodLeakAccepted q.run K H D L Kₒ S lay k δ decode reg Dj n} (W : WindowRuns R)
    (h2 : PeriodRunsV2 W fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (h4 : WindowCoinsLive W)
    (h5 : SessionIsSampledRun W (couplingGuard q))
    (h6 : NetworkCouples R (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) W)
    (hρ : WindowValuesRegistered R fun ξ => (V.outcome θ F c σ q.coins ξ).sent)
    (hreg : WindowRegistersAs W (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (couplingGuard q))
    {Brk : ((i : q.I) → q.Ω i) → Prop} (hcov : RecordCoversOff W (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) Brk)
    (hbrk : ∀ ξ, Brk ξ → q.brk ξ) :
    Subprotocol.NetworkWindowLe R q.verdict :=
  networkWindowLe_off (fun _ h => h) W h2 h4 h5 h6 hρ hreg hcov fun _ hg hb => hg.2 (hbrk _ hb)

/-! ## The clock form -/

/-- **The ρ gap where the registered windows register no values.** Where onsite's network component accepts, window
`i`'s statement is registered as `sw i` at every coin (S1's program check, S2, S6's window count and
`WindowRegistersAs` unguarded), so it registers what `sw i` registers: nothing, at the clock form, whose execution
phase's Program reads no registered input, and at any window whose registration names none. -/
theorem windowValuesRegistered_of_sw (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} (h2 : PeriodRunsV2 W s)
    (h6 : NetworkCouples R s W) (hreg : WindowRegistersAs W s fun _ => True) (hsw : ∀ i, reg (R.sw i) = ∅) :
    WindowValuesRegistered R s :=
  windowValuesRegistered_of_empty fun ξ i st hca hst => by
    obtain ⟨p, c, e, hnp, hacc⟩ := componentAccepts_network_windows periodAcceptMeans hca
    obtain ⟨rg, hrg, hprog⟩ := windowProgram_of_accepted hacc (lt_of_lt_of_le i.isLt (h6.windows ξ p c e hnp))
    rw [(hreg ξ i p c e rg st trivial (h2 ξ i hca) hnp hrg hprog hst).2.2.2.2.1, hsw i]

/-- **The clock form's record covers the horizon off the break**: the record in the clock form (`RecordsClock`), the
period's certificate check (S1, at S6's window count), and the device certifying with its issuer (`LogIssued`). -/
theorem recordCoversOff_break_clock (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} (h6 : NetworkCouples R s W)
    (F : ClockFormat R.Conn R.Cell L Kₒ S) (hF : RecordsClock R F) {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : I) → Ω i) → DeviceOutput} (sess : ((i : I) → Ω i) → ByteArray)
    (hlog : LogIssued s dir out fun ξ (i : Fin n) c => F.status (W.sched ξ) (R.arr (W.srv ξ)) i.val c = .complete) :
    RecordCoversOff W s (CertBreak s sess out) :=
  (recordCoversOff_of_certified W s F _ hF (certifiedOnAccept_period dir out h6.windows)
    (signedComplete_period W F hlog)).mono fun _ h => certBreak_of_forged sess hlog h

/-- **The network certifier's per-window bound at the clock form**, from S2, S4–S6 and `WindowRegistersAs` at every
coin: `sw i` registering no values and the record covering the horizon off the break (`recordCoversOff_break_clock`),
so the bound gains the break's probability; `windowLe_off` at the same premises, at a guard that excludes the break,
bounds the acceptance by `PrBad` alone. -/
theorem windowLe_clock (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} {G : ((i : I) → Ω i) → Prop}
    (h2 : PeriodRunsV2 W s) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G) (h6 : NetworkCouples R s W)
    (hreg : WindowRegistersAs W s fun _ => True) (hsw : ∀ i, reg (R.sw i) = ∅)
    (F : ClockFormat R.Conn R.Cell L Kₒ S) (hF : RecordsClock R F) {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : I) → Ω i) → DeviceOutput} (sess : ((i : I) → Ω i) → ByteArray)
    (hlog : LogIssued s dir out fun ξ (i : Fin n) c => F.status (W.sched ξ) (R.arr (W.srv ξ)) i.val c = .complete) :
    ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      ComponentAccepts .onsite .network (s (Function.update ξ (r.own .network) x)) ∧
        G (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt (Function.update ξ (r.own .network) x) i)
          (R.wit (Function.update ξ (r.own .network) x) i) (R.obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V (Rlt R) (E D decode) (R.stmt ξ i) (R.prover ξ i) +
      Pouw.pr (fun x : Ω (r.own .network) => CertBreak s sess out (Function.update ξ (r.own .network) x)) :=
  windowLeOff W h2 h4 h5 h6 (windowValuesRegistered_of_sw W h2 h6 hreg hsw) (hreg.mono fun _ _ => trivial)
    (recordCoversOff_break_clock W h6 F hF sess hlog)

/-- **The refinement in the game at the clock form**: at a coupling whose break holds `CertBreak`, the network
certifier's refinement at its verdict with S8's coverage not a premise. The ρ gap stays one: the coupling's statement
is registered as `sw i` only under its guard, so `windowValuesRegistered_of_sw` doesn't apply. -/
theorem networkWindowLe_verdict_clock {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (q : V.Coupling .onsite .network θ F c σ) {K : ℝ}
    {R : RealizesCommOfPeriodLeakAccepted q.run K H D L Kₒ S lay k δ decode reg Dj n} (W : WindowRuns R)
    (h2 : PeriodRunsV2 W fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (h4 : WindowCoinsLive W)
    (h5 : SessionIsSampledRun W (couplingGuard q))
    (h6 : NetworkCouples R (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) W)
    (hρ : WindowValuesRegistered R fun ξ => (V.outcome θ F c σ q.coins ξ).sent)
    (hreg : WindowRegistersAs W (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (couplingGuard q))
    (Fm : ClockFormat R.Conn R.Cell L Kₒ S) (hF : RecordsClock R Fm) {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : q.I) → q.Ω i) → DeviceOutput} (sess : ((i : q.I) → q.Ω i) → ByteArray)
    (hlog : LogIssued (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) dir out
      fun ξ (i : Fin n) c => Fm.status (W.sched ξ) (R.arr (W.srv ξ)) i.val c = .complete)
    (hbrk : ∀ ξ, CertBreak (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) sess out ξ → q.brk ξ) :
    Subprotocol.NetworkWindowLe R q.verdict :=
  networkWindowLe_verdict_off q W h2 h4 h5 h6 hρ hreg (recordCoversOff_break_clock W h6 Fm hF sess hlog) hbrk

/-! ## The rows form -/

/-- **The rows form (the scan's record).** The realization's record format is the rows form `F` (`record`); each window
is its registered horizon's buckets (`horizon`); and every bucket's jitter class in the format is below `Dj`
(`jitter`: J, `JitterBounded`, at the format's jitter class rather than a `Certifier`'s observation). -/
structure RecordsRows (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n)
    (F : RowsFormat R.Conn R.Cell L Kₒ) : Prop where
  record : ∀ sc a v, R.fmt.record sc a v = F.record S n sc a v
  horizon : ∀ i, F.T = k (R.sw i)
  jitter : ∀ sc a b, F.jitter sc a b < Dj

/-- **The rows form's record covers the horizon off the forger's event**: the format (`RecordsRows`), the period's
certificate check (`CertifiedOnAccept`), and the device certifying a boundary link's transfer blocks only where every
bucket of the window has that link's status complete. -/
theorem recordCoversOff_rows (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) (F : RowsFormat R.Conn R.Cell L Kₒ)
    {Cert : Type} (C : CertRuns R Cert) (hF : RecordsRows R F) (hacc : CertifiedOnAccept s C)
    (hB : ∀ ξ (i : Fin n) c x, C.accepted ξ i c x → C.signed ξ x →
      ∀ t < F.T, F.status (W.sched ξ) (R.arr (W.srv ξ)) (i.val * F.T + t) c = .complete) :
    RecordCoversOff W s (CertForged C) := by
  intro ξ i hca hnb
  simp only [WindowRuns.record, hF.record, RowsFormat.record, ← hF.horizon i]
  refine ⟨fun t _ ht c => ?_, by simp, fun y hy => ?_⟩
  · simp only [List.getElem_map, List.getElem_range]
    obtain ⟨x, hx⟩ := hacc ξ i c hca
    exact hB ξ i c x hx (not_not.1 fun hns => hnb ⟨i, c, x, hx, hns⟩) t (by simpa using ht)
  · obtain ⟨t, -, rfl⟩ := List.mem_map.1 hy
    exact hF.jitter _ _ _

/-- **The rows form's record covers the horizon off the break**: the record in the rows form (`RecordsRows`), the
period's certificate check (S1, at S6's window count), and the device certifying with its issuer (`LogIssued`, a
complete record of a link's transfer blocks being every bucket of the window complete). -/
theorem recordCoversOff_break_rows (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} (h6 : NetworkCouples R s W)
    (F : RowsFormat R.Conn R.Cell L Kₒ) (hF : RecordsRows R F) {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : I) → Ω i) → DeviceOutput} (sess : ((i : I) → Ω i) → ByteArray)
    (hlog : LogIssued s dir out fun ξ (i : Fin n) c =>
      ∀ t < F.T, F.status (W.sched ξ) (R.arr (W.srv ξ)) (i.val * F.T + t) c = .complete) :
    RecordCoversOff W s (CertBreak s sess out) :=
  (recordCoversOff_rows W s F _ hF (certifiedOnAccept_period dir out h6.windows) (signed_complete hlog)).mono
    fun _ h => certBreak_of_forged sess hlog h

/-- **The network certifier's per-window bound at the rows form**, from S2, S4–S6, the ρ gap and `WindowRegistersAs`:
the record covering the horizon off the break (`recordCoversOff_break_rows`), so the bound gains the break's
probability; `windowLe_off` at the same premises, at a guard that excludes the break, bounds the acceptance by `PrBad`
alone. -/
theorem windowLe_rows (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} {G : ((i : I) → Ω i) → Prop}
    (h2 : PeriodRunsV2 W s) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G) (h6 : NetworkCouples R s W)
    (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s G)
    (F : RowsFormat R.Conn R.Cell L Kₒ) (hF : RecordsRows R F) {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : I) → Ω i) → DeviceOutput} (sess : ((i : I) → Ω i) → ByteArray)
    (hlog : LogIssued s dir out fun ξ (i : Fin n) c =>
      ∀ t < F.T, F.status (W.sched ξ) (R.arr (W.srv ξ)) (i.val * F.T + t) c = .complete) :
    ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      ComponentAccepts .onsite .network (s (Function.update ξ (r.own .network) x)) ∧
        G (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt (Function.update ξ (r.own .network) x) i)
          (R.wit (Function.update ξ (r.own .network) x) i) (R.obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V (Rlt R) (E D decode) (R.stmt ξ i) (R.prover ξ i) +
      Pouw.pr (fun x : Ω (r.own .network) => CertBreak s sess out (Function.update ξ (r.own .network) x)) :=
  windowLeOff W h2 h4 h5 h6 hρ hreg (recordCoversOff_break_rows W h6 F hF sess hlog)

/-- **The refinement in the game at the rows form**: at a coupling whose break holds `CertBreak`, the network
certifier's refinement at its verdict with S8's coverage not a premise; the ρ gap stays one. -/
theorem networkWindowLe_verdict_rows {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (q : V.Coupling .onsite .network θ F c σ) {K : ℝ}
    {R : RealizesCommOfPeriodLeakAccepted q.run K H D L Kₒ S lay k δ decode reg Dj n} (W : WindowRuns R)
    (h2 : PeriodRunsV2 W fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (h4 : WindowCoinsLive W)
    (h5 : SessionIsSampledRun W (couplingGuard q))
    (h6 : NetworkCouples R (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) W)
    (hρ : WindowValuesRegistered R fun ξ => (V.outcome θ F c σ q.coins ξ).sent)
    (hreg : WindowRegistersAs W (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) (couplingGuard q))
    (Fm : RowsFormat R.Conn R.Cell L Kₒ) (hF : RecordsRows R Fm) {dir : Fin Kₒ → NetworkCheck.Direction}
    {out : ((i : q.I) → q.Ω i) → DeviceOutput} (sess : ((i : q.I) → q.Ω i) → ByteArray)
    (hlog : LogIssued (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) dir out fun ξ (i : Fin n) c =>
      ∀ t < Fm.T, Fm.status (W.sched ξ) (R.arr (W.srv ξ)) (i.val * Fm.T + t) c = .complete)
    (hbrk : ∀ ξ, CertBreak (fun ξ => (V.outcome θ F c σ q.coins ξ).sent) sess out ξ → q.brk ξ) :
    Subprotocol.NetworkWindowLe R q.verdict :=
  networkWindowLe_verdict_off q W h2 h4 h5 h6 hρ hreg (recordCoversOff_break_rows W h6 Fm hF sess hlog) hbrk

end Verity.Network
