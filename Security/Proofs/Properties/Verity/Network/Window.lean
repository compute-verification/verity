import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted
import Definitions.NetworkCertifier.RecordRows
import Properties.Verity.Network
import Proofs.Properties.NetworkCertifier.Period
import Proofs.Properties.Verity.Network.Coupling

/-!
# The network certifier's per-window bound at its compiled component, from named steps

`Verity.Subprotocol.NetworkWindowLe R A` (`window_le` at the verdict `A`) bounds, at each window `i` and
each value `ξ` of the other coins, the probability over the network audit's own coin that `A` holds while window `i`
fails its good event (`NetTiming.WindowGood`) by the sampled verifier's `PrBad` at the leak relation at the network
certifier's rows (`Rlt`). `windowLe` proves it with `A` onsite's network component accepting what the prover sent,
from these steps (`Verity.Network.networkWindowLe` takes it to any `A` that implies that acceptance, a coupling's
verdict at the code `Vocab.Coupling.verdict` among them):

| step | content | state |
| --- | --- | --- |
| S1 | an accepted period accepted every link-window (`periodAcceptMeans`) | proved |
| S2 | Period runs one-stage v2 on each window's case (`PeriodRunsV2`) | open |
| S3 | v2's acceptance is `--zk`'s on every session (`V2AcceptMeans`) | open |
| S4 | the sessions' coins are the live coins (`WindowCoinsLive`) | open |
| S5 | the accepted sessions are an accepting sampled run, under a guard (`SessionIsSampledRun`) | open |
| S6 | the realization is the compiled run's (`NetworkCouples`) | open |
| S7 | a uniform coin map's event is below the run's (`pr_le_of_coins`) | proved |
| S8 | `WindowGood` past the leak relation, on acceptance (`windowGood_of_accepts`) | proved, from premises |
| ρ | the trace holds the registered values (`WindowValuesRegistered`) | proved where no window registers values (`windowValuesRegistered_of_empty`); else open |

S8 rests on S1, S2, S6 and `trusted-host` (`R.trusted`: the record is the device's code's, `obs_eq_record`), the ρ gap,
and two premises: the window's statement registered as `sw i` (`WindowRegistersAs`: one-stage's registration
checks), and the network certifier's record covering the horizon (`RecordCovers`, open). For the
rows format that is two facts per bucket (`recordCovers_of_rows`); for one-stage v2's clock form it is the window's
statuses alone (`recordCovers_of_clock`), which N2's head check gives only off its soundness error and up to a forgery
under the device's key (`ClockStatusComplete`). With the device certifying only complete records (B), the period's
in-the-clear certificate check gives it off the forger's event (`recordCoversOff_of_certified`), and the bound gains
that event's probability (`windowLeOff`).

The realization's fields the steps read beyond `RealizesCommOfPeriodLeakAccepted`'s are `WindowRuns`', in the
`SampledSession` form: each a property of the deployed run, which the named steps tie to the compiled component.
-/

namespace Verity.Network

open Accounting Accounting.JointBridge Claims Claims.Sampling NetTiming Properties.NetworkCertifier

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
  [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin Col T} {L Kₒ S : ℕ} {lay : Stmt → Layout L S}
  {k : Stmt → ℕ} {δ : ℝ} {decode : List (ℕ × Col) → List T} {reg : Stmt → Finset ℕ} {Dj n : ℕ}
  {r : PeriodRun d α C Ω} {K : ℝ}

/-- The leak relation at the network certifier's rows that `NetworkWindowLe`'s `PrBad` reads, with the registered
inputs pinned. -/
abbrev Rlt (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) : Rel Wit :=
  RLeakTap H D.commit D.parse D.partition L Kₒ S lay k δ (fun m => AsReg m.prog (lay m) (k m) (reg m)) Dj R.tops

/-- **The network audit's runs at each window**: what the named steps read of the deployed run at a realization `R`,
beyond its fields.

* `v2 ξ i`: one-stage v2 accepts window `i`'s case at the coins `ξ` (S2's conclusion, S3's hypothesis);
* `session ξ i`: flock-verify `--zk` accepts every session of window `i`'s one-stage proof at `ξ` (S3's conclusion);
* `coins ξ i x`: the tape of window `i`'s prover and the sampled verifier's coins that the network audit's own coin `x`
  gives, at the other coins `ξ` (S4, S5);
* `sched ξ`, `srv ξ`: the network certifier's schedule and the server's state at `ξ`, which its device runs on (S6). -/
structure WindowRuns (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) where
  v2 : ((i : I) → Ω i) → Fin n → Prop
  session : ((i : I) → Ω i) → Fin n → Prop
  coins : (ξ : (i : I) → Ω i) → (i : Fin n) → Ω (r.own .network) → R.Tape ξ i × (Fin (D.V.k (R.stmt ξ i)) → Coin)
  sched : ((i : I) → Ω i) → WardenSchedule R.Conn
  srv : ((i : I) → Ω i) → R.Srv

variable {R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n}

/-- The network certifier device's code's record of window `i` at the coins `ξ`, on its schedule and the server's state
there. -/
def WindowRuns.record (W : WindowRuns R) (ξ : (i : I) → Ω i) (i : Fin n) : List (Obs L Kₒ S) :=
  R.fmt.record (W.sched ξ) (R.arr (W.srv ξ)) (wardenCode (W.sched ξ) (R.arr (W.srv ξ))) i

/-! ## The named steps -/

/-- **S2 (open): the period check runs one-stage v2 on each window's case.** Where onsite's
network component accepts at `ξ`, one-stage v2 accepted window `i`'s case. Today's `NetworkCheck.Period` reads each
window's one-stage answer from the evidence the prover sends (`PeriodWindowProved`, `PeriodRowsProved`) and runs no
verifier on it, so this needs Period to run `OneStageZk.checkBlobs` on each window's case, over the core interface's
blobs and the blob product. -/
def PeriodRunsV2 (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) : Prop :=
  ∀ ξ i, ComponentAccepts .onsite .network (s ξ) → W.v2 ξ i

/-- **S3: v2's acceptance is flock-verify `--zk`'s on every session**, at window `i`'s case:
`Properties.SampledProofs.OneStageZk.AcceptMeans`. -/
def V2AcceptMeans (W : WindowRuns R) : Prop :=
  ∀ ξ i, W.v2 ξ i → W.session ξ i

/-- **S4: the sessions' coins are the challenger's live coins.** At each window and each value
of the other coins, the network audit's own uniform coin gives the prover's tape and the sampled verifier's coins
uniformly. -/
def WindowCoinsLive (W : WindowRuns R) : Prop :=
  ∀ ξ i (q : R.Tape ξ i × (Fin (D.V.k (R.stmt ξ i)) → Coin) → Prop), Pouw.pr (fun x => q (W.coins ξ i x)) = Claims.Pr q

/-- **S5 (the one-stage twin of `Verity.Recursive.acc_of_accepts`): the accepted sessions are an
accepting run of the sampled verifier, where the guard `G` holds.** Where `G` holds and every session of window `i`'s
proof is accepted at the coins `ξ` with the network audit's own coin `x`, the sampled verifier's run at the window's
statement, against its prover at the tape and coins `x` gives, accepts. The guard is what the run may need beyond
acceptance: at a coupling's verdict at the code (`Vocab.Coupling.verdict`), each release its check read at its round
of the device's log and the coupling's break, the events its finder answers for, absent. `fun _ => True` is the
unguarded form. Were the bridge restated over C-Flock's game instead, the `SampledSession` coupling would give this,
with the bridge's KS assumption changed: a claims change. -/
def SessionIsSampledRun (W : WindowRuns R) (G : ((i : I) → Ω i) → Prop) : Prop :=
  ∀ ξ i x, G (Function.update ξ (r.own .network) x) → W.session (Function.update ξ (r.own .network) x) i →
    (Claims.run H (R.prover ξ i) D.V (R.stmt ξ i) (W.coins ξ i x).1 (W.coins ξ i x).2).out = .acc

/-- **S6 (the bridge): the realization is the compiled run's**, in the `SampledSession.Couples` form:

* `stmt`: window `i`'s statement is fixed before the network audit's own coin (its registration);
* `wit`: the realization's witness is in the leak relation wherever the window's statement has a witness there;
* `obs`: the realization's record of the windows is the network certifier device's, on its schedule and the server's
  state;
* `windows`: the period onsite's network case decodes to has the realization's `n` windows at least. -/
structure NetworkCouples (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n)
    (s : ((i : I) → Ω i) → Sent) (W : WindowRuns R) : Prop where
  stmt : ∀ ξ i x, R.stmt (Function.update ξ (r.own .network) x) i = R.stmt ξ i
  wit : ∀ ξ i, (∃ w, Rlt R (R.stmt ξ i) w) → Rlt R (R.stmt ξ i) (R.wit ξ i)
  obs : ∀ ξ, R.obs ξ = (R.run R.device (W.sched ξ, W.srv ξ)).2
  windows : ∀ ξ p c e, NetworkPeriod .onsite (s ξ) p c e → n ≤ p.windows

/-- **The ρ gap (open): the trace holds the registered values at the registered
inputs.** Where onsite's network component accepts at `ξ` and the realization's witness is in the leak relation, the
trace it opens holds window `i`'s registered values `ρ i` at the registered inputs: the gap `PeriodLeakAccepted` names,
which the registration's commitment binds (`Commitment.Binds`) once `ρ i` is what it committed. -/
def WindowValuesRegistered (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n)
    (s : ((i : I) → Ω i) → Sent) : Prop :=
  ∀ ξ i st, ComponentAccepts .onsite .network (s ξ) → D.parse (R.stmt ξ i) = some st → Rlt R (R.stmt ξ i) (R.wit ξ i) →
    ∀ g ∈ reg st, traceVal st.prog ((R.wit ξ i).val H D.commit st) g = R.ρ i g

/-- **The ρ gap where the windows register no values.** Where every statement of a window that onsite's network
component accepts has no registered inputs, the trace holds `ρ i` at each of them vacuously. A clock-form link-window is
such a window: N2's one-stage parameters name no registered values (`registered` and `proofs.registered` null,
`n2_zk`'s), and one-stage v2 refuses a Program with a `RegisteredInput` under them (`OneStageZk.run`). A window whose
Program registers values (weights pinned before the period) needs the registration's openings instead, at the cost of
`B_reg` (`regColl_of_disagree`). -/
theorem windowValuesRegistered_of_empty {s : ((i : I) → Ω i) → Sent}
    (h : ∀ ξ i st, ComponentAccepts .onsite .network (s ξ) → D.parse (R.stmt ξ i) = some st → reg st = ∅) :
    WindowValuesRegistered R s := by
  intro ξ i st hca hst _ g hg
  simp [h ξ i st hca hst] at hg

/-- **S8's registration premise (one-stage's registration checks, R1–R6, at the window's case).** Where v2
accepted window `i`'s case at `ξ` and the period onsite's network case decodes to registers window `i` under a
registration naming the period's Program (the one field of it the period check reads, S1's `PeriodWindowProgram`),
the window's statement is registered as `sw i`: the same program, design, layout, horizon, registered inputs and
partition. -/
def WindowRegistersAs (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) : Prop :=
  ∀ ξ (i : Fin n) p c e rg st, W.v2 ξ i → NetworkPeriod .onsite (s ξ) p c e → c.windows[i.val]? = some rg →
    PeriodWindowProgram p rg → D.parse (R.stmt ξ i) = some st → SameRegistration D lay k reg (R.sw i) st

/-- **S8's coverage premise (open).** Where onsite's network component accepts at `ξ`, the network
certifier's record of window `i` covers its registered horizon `k (sw i)`: that many steps, each with every status
complete and its jitter class below `Dj`. Today's period check reads neither statuses nor jitter. For the rows format
it is two facts per bucket (`recordCovers_of_rows`); at one-stage v2's clock form, whose record is untimed and whose
statuses N2's head check reads, it is the window's statuses alone (`recordCovers_of_clock`). -/
def RecordCovers (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) : Prop :=
  ∀ ξ i, ComponentAccepts .onsite .network (s ξ) →
    Covered (k (R.sw i)) (W.record ξ i) ∧ (W.record ξ i).length = k (R.sw i) ∧ ∀ y ∈ W.record ξ i, y.jitter < Dj

/-! ## S8: `WindowGood` past the leak relation, on acceptance -/

/-- An accepted period's link-window `i < p.windows` is registered, under a registration naming the period's Program. -/
theorem windowProgram_of_accepted {p : NetworkCheck.Period.Params} {c : NetworkCheck.Period.Commitment}
    {e : NetworkCheck.Period.Evidence} (h : PeriodWindowsAccepted p c e) {i : ℕ} (hi : i < p.windows) :
    ∃ rg, c.windows[i]? = some rg ∧ PeriodWindowProgram p rg := by
  obtain ⟨hc, he, hall⟩ := h
  have hc' : i < c.windows.size := by omega
  have he' : i < e.windows.size := by omega
  refine ⟨c.windows[i], Array.getElem?_eq_getElem hc', ?_⟩
  have hw := hall i c.windows[i] e.windows[i] (Array.getElem?_eq_getElem hc') (Array.getElem?_eq_getElem he')
  split at hw
  · exact hw.1.1
  · exact hw.1

/-- **Under `trusted-host`, the realization's record is the device's code's**: S6's `obs` and `R.trusted`. -/
theorem obs_eq_record (W : WindowRuns R) (h : ∀ ξ, R.obs ξ = (R.run R.device (W.sched ξ, W.srv ξ)).2)
    (ξ : (i : I) → Ω i) (i : Fin n) : R.obs ξ i = W.record ξ i := by
  rw [h, R.trusted]
  rfl

/-- **The rows format's record covers the horizon** where each window is the horizon's buckets (`F.T = k (sw i)`) and,
on acceptance, every bucket of the window has every status complete and its jitter class below `Dj`. -/
theorem recordCovers_of_rows (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) (F : RowsFormat R.Conn R.Cell L Kₒ)
    (hfmt : ∀ sc a v, R.fmt.record sc a v = F.record S n sc a v) (hT : ∀ i, F.T = k (R.sw i))
    (hst : ∀ ξ (i : Fin n), ComponentAccepts .onsite .network (s ξ) → ∀ t < F.T, ∀ κ,
      F.status (W.sched ξ) (R.arr (W.srv ξ)) (i.val * F.T + t) κ = .complete)
    (hj : ∀ ξ (i : Fin n), ComponentAccepts .onsite .network (s ξ) → ∀ t < F.T,
      F.jitter (W.sched ξ) (R.arr (W.srv ξ)) (i.val * F.T + t) < Dj) :
    RecordCovers W s := by
  intro ξ i hca
  simp only [WindowRuns.record, hfmt, RowsFormat.record, ← hT i]
  refine ⟨fun t _ ht κ => ?_, by simp, fun y hy => ?_⟩
  · simp only [List.getElem_map, List.getElem_range]
    exact hst ξ i hca t (by simpa using ht) κ
  · obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hy
    exact hj ξ i hca t (List.mem_range.1 ht)

/-! ## The clock form's record -/

/-- **The clock form's record format** (one-stage v2's link-window, `n2_zk`'s): window `i` is `T` steps of the
declared grid, each step its bucket's rows, under one status per boundary link for the whole window (the record head's
one status, W5), and no timing: the record is untimed, so each step's jitter class is 0. -/
structure ClockFormat (κ Cell : Type) (L K S : ℕ) where
  T : ℕ
  rows : WardenSchedule κ → (κ → Arrivals Cell) → (κ → List (ℕ × CellView Cell)) → ℕ → Fin L → Row S (List Bool)
  status : WardenSchedule κ → (κ → Arrivals Cell) → ℕ → Fin K → Status

namespace ClockFormat

variable {κ Cell : Type} {L K S : ℕ} (F : ClockFormat κ Cell L K S)

/-- **The record** over `n` windows: window `i`'s step `t` is bucket `i·T + t`'s rows, window `i`'s statuses and
jitter class 0. -/
def record (n : ℕ) (s : WardenSchedule κ) (a : κ → Arrivals Cell) (v : κ → List (ℕ × CellView Cell)) :
    Fin n → List (Obs L K S) :=
  fun i => (List.range F.T).map fun t =>
    { rows := F.rows s a v (i.val * F.T + t), jitter := 0, status := F.status s a i.val }

/-- **A clock-form window covers the horizon `k`** where it is `k` steps, its statuses are complete and the jitter
bound `Dj` is at least 1, above the untimed record's 0. -/
theorem record_covers {n : ℕ} {s : WardenSchedule κ} {a : κ → Arrivals Cell} {v : κ → List (ℕ × CellView Cell)}
    (i : Fin n) {k Dj : ℕ} (hT : F.T = k) (hst : ∀ c, F.status s a i.val c = .complete) (hDj : 1 ≤ Dj) :
    Covered k (F.record n s a v i) ∧ (F.record n s a v i).length = k ∧ ∀ y ∈ F.record n s a v i, y.jitter < Dj := by
  subst hT
  refine ⟨fun t _ _ c => ?_, by simp [record], fun y hy => ?_⟩
  · simp [record, hst]
  · obtain ⟨t, _, rfl⟩ := List.mem_map.1 hy
    exact hDj

end ClockFormat

/-- **The clock form (the deployed record).** The realization's record format is the clock form
`F` (`record`); each window is its registered horizon's steps (`horizon`: Period registers every window at the declared
grid, whose T the network certifier runs, and N2's head check holds each head's T to that grid's); and the jitter bound
is at least 1 (`jitter`: Period v2's `paramsOf` refuses a declared jitter below 1), above the untimed record's 0. -/
structure RecordsClock (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n)
    (F : ClockFormat R.Conn R.Cell L Kₒ S) : Prop where
  record : ∀ sc a v, R.fmt.record sc a v = F.record n sc a v
  horizon : ∀ i, F.T = k (R.sw i)
  jitter : 1 ≤ Dj

/-- **The clock form's statuses (open: N2's soundness and the device's key).** Where onsite's network component
accepts at `ξ`, every boundary link's status in the device's clock-form record of window `i` is complete. N2's head
check accepts a head only at status complete and the declared grid (`head_check.NetworkHeadFields`), but it runs inside
N2's proof, on the head N2 commits, and reaches the device's record only through the certificate, whose message names
the link-window in the clear and signs the head's salted record digest (`certificate.certificate_message`). The device
signs a head of any status (`Issuer.certify`), and `trusted-host` doesn't keep its key secret. So acceptance gives this
only off N2's soundness error (S3–S5 take accepted sessions to an accepting sampled run, not to a witness the head check
holds on), up to a SHA-512 collision between the committed head and the device's, and up to a forgery under the
device's key. Once the device refuses to certify an incomplete status (B), it holds off the forger's event alone
(`recordCoversOff_of_certified`, `windowLeOff`). -/
def ClockStatusComplete (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) (F : ClockFormat R.Conn R.Cell L Kₒ S) :
    Prop :=
  ∀ ξ (i : Fin n), ComponentAccepts .onsite .network (s ξ) → ∀ c, F.status (W.sched ξ) (R.arr (W.srv ξ)) i.val c = .complete

/-- **The clock form's record covers the horizon** (S8's coverage premise at one-stage v2's record): each window is the
horizon's steps with jitter class 0 below the bound (`RecordsClock`), so it covers the horizon wherever its statuses
are complete (`ClockStatusComplete`). -/
theorem recordCovers_of_clock (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) (F : ClockFormat R.Conn R.Cell L Kₒ S)
    (hF : RecordsClock R F) (hst : ClockStatusComplete W s F) : RecordCovers W s := by
  intro ξ i hca
  simp only [WindowRuns.record, hF.record]
  exact F.record_covers i (hF.horizon i) (hst ξ i hca) hF.jitter

/-! ## B: the device certifies only a complete record

With the device refusing to certify an incomplete status (`Issuer.certify`), the statuses follow from the period's
in-the-clear certificate check and the device's log of what it certified, off one event: the period accepted a
certificate the device never certified (`CertForged`). The per-window bound then gains that event's probability
(`windowLeOff`), which `euf-cma/lms-sha512-n64` and `ecr/sha-512` bound outside the theorem, each on its own fixed
finder. -/

/-- **The network audit's certificates at the coins `ξ`**: `accepted ξ i c x`, the period accepted the certificate `x`
as window `i`'s of boundary link `c` (`NetworkCheck.Period.certifies`: its fields name that link, its end and the window,
and it verifies under LMS over `certificate_message` under the parameters' key in their session); `signed ξ x`, the
network certifier device certified `x` in its run (its log of the certificates `Issuer.certify` issued). -/
structure CertRuns (R : RealizesCommOfPeriodLeakAccepted r K H D L Kₒ S lay k δ decode reg Dj n) (Cert : Type) where
  accepted : ((i : I) → Ω i) → Fin n → Fin Kₒ → Cert → Prop
  signed : ((i : I) → Ω i) → Cert → Prop

/-- **The period checks each window's certificates in the clear** (S1 at one-stage v2's period, with D:
`Period.windowCheck` refuses a link-window unless both its certificates, egress then ingress, are the parameters'
key's, name the link, their end and the window, and verify in the session). Where onsite's network component accepts
at `ξ`, the period accepted a certificate of every boundary link of every window. -/
def CertifiedOnAccept (s : ((i : I) → Ω i) → Sent) {Cert : Type} (C : CertRuns R Cert) : Prop :=
  ∀ ξ (i : Fin n) c, ComponentAccepts .onsite .network (s ξ) → ∃ x, C.accepted ξ i c x

/-- **B: the device certifies only a complete record, under `trusted-host`** (`Issuer.certify`
lists an incomplete record's link-window in `refused` and signs nothing for it). Where the period accepted `x` as window
`i`'s certificate of boundary link `c` and the device certified `x`, the device certified its record of that
link-window, so that link's status in its record of window `i` is complete. -/
def SignedComplete (W : WindowRuns R) (F : ClockFormat R.Conn R.Cell L Kₒ S) {Cert : Type} (C : CertRuns R Cert) :
    Prop :=
  ∀ ξ (i : Fin n) c x, C.accepted ξ i c x → C.signed ξ x →
    F.status (W.sched ξ) (R.arr (W.srv ξ)) i.val c = .complete

/-- **The forger's event** at `ξ`: the period accepted a certificate the device never certified. Its fixed finder
outputs the first such certificate: an LMS forgery under the device's key over its message (`euf-cma/lms-sha512-n64`),
or, where the device signed that message for another certificate's fields, two `certificate_message` inputs with one
SHA-512 digest (`ecr/sha-512`). Both claims apply outside the theorem, each to its own fixed finder. -/
def CertForged {Cert : Type} (C : CertRuns R Cert) (ξ : (i : I) → Ω i) : Prop :=
  ∃ (i : Fin n) (c : Fin Kₒ) (x : Cert), C.accepted ξ i c x ∧ ¬ C.signed ξ x

/-- **S8's coverage premise off an event `Brk`**: `RecordCovers` where `Brk` doesn't hold at `ξ`. -/
def RecordCoversOff (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent) (Brk : ((i : I) → Ω i) → Prop) : Prop :=
  ∀ ξ i, ComponentAccepts .onsite .network (s ξ) → ¬ Brk ξ →
    Covered (k (R.sw i)) (W.record ξ i) ∧ (W.record ξ i).length = k (R.sw i) ∧ ∀ y ∈ W.record ξ i, y.jitter < Dj

/-- **Under B the clock form's record covers the horizon off the forger's event**: the format (`RecordsClock`), the
period's certificate check (`CertifiedOnAccept`) and the device certifying only complete records (`SignedComplete`). -/
theorem recordCoversOff_of_certified (W : WindowRuns R) (s : ((i : I) → Ω i) → Sent)
    (F : ClockFormat R.Conn R.Cell L Kₒ S) {Cert : Type} (C : CertRuns R Cert) (hF : RecordsClock R F)
    (hacc : CertifiedOnAccept s C) (hB : SignedComplete W F C) : RecordCoversOff W s (CertForged C) := by
  intro ξ i hca hnb
  simp only [WindowRuns.record, hF.record]
  refine F.record_covers i (hF.horizon i) (fun c => ?_) hF.jitter
  obtain ⟨x, hx⟩ := hacc ξ i c hca
  exact hB ξ i c x hx (not_not.1 fun hns => hnb ⟨i, c, x, hx, hns⟩)

/-- **S8 at one `ξ`**: where the component accepts, the realization's witness is in the leak relation and window `i`'s
record covers the horizon, window `i` is good: in scope (`R.scope`), registered as `sw i` (S1's program check through
`WindowRegistersAs`, at S2's v2 and S6's window count), the trace holding `ρ i` (`WindowValuesRegistered`), and the
record the device's code's under `trusted-host`. -/
theorem windowGood_of_covers (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} (h2 : PeriodRunsV2 W s)
    (h6 : NetworkCouples R s W) (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s)
    {ξ : (i : I) → Ω i} {i : Fin n} (hca : ComponentAccepts .onsite .network (s ξ))
    (hcov : Covered (k (R.sw i)) (W.record ξ i) ∧ (W.record ξ i).length = k (R.sw i) ∧
      ∀ y ∈ W.record ξ i, y.jitter < Dj)
    (hw : Rlt R (R.stmt ξ i) (R.wit ξ i)) :
    WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt ξ i) (R.wit ξ i) (R.obs ξ i) := by
  obtain ⟨_, st, hst, _⟩ := R.scope ξ i
  obtain ⟨p, c, e, hnp, hacc⟩ := componentAccepts_network_windows periodAcceptMeans hca
  obtain ⟨rg, hrg, hprog⟩ := windowProgram_of_accepted hacc (lt_of_lt_of_le i.isLt (h6.windows ξ p c e hnp))
  have hsame := hreg ξ i p c e rg st (h2 ξ i hca) hnp hrg hprog hst
  obtain ⟨hcv, hlen, hj⟩ := hcov
  rw [obs_eq_record W h6.obs]
  refine ⟨hw, R.scope ξ i, st, hst, hsame, hρ ξ i st hca hst hw, ?_, ?_, hj⟩
  · rw [hsame.2.2.2.1]; exact hcv
  · rw [hsame.2.2.2.1]; exact hlen

/-- **S8: what onsite's network component accepting gives of window `i`'s good event past the leak relation.** At
`ξ` where the component accepts and the realization's witness is in the leak relation, window `i` is good
(`windowGood_of_covers`), its record covering the horizon (`RecordCovers`). -/
theorem windowGood_of_accepts (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} (h2 : PeriodRunsV2 W s)
    (h6 : NetworkCouples R s W) (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s)
    (hcov : RecordCovers W s) {ξ : (i : I) → Ω i} {i : Fin n} (hca : ComponentAccepts .onsite .network (s ξ))
    (hw : Rlt R (R.stmt ξ i) (R.wit ξ i)) :
    WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt ξ i) (R.wit ξ i) (R.obs ξ i) :=
  windowGood_of_covers W h2 h6 hρ hreg hca (hcov ξ i hca) hw

/-! ## The per-window bound -/

theorem exists_of_ext {W : Type} {Q : Rel W} {x : Msg} : ∀ {o : Option W}, Q.ext x o → ∃ w, Q x w
  | some w, h => ⟨w, h⟩
  | none, h => h.elim

/-- An event is no likelier than its part off `B` plus `B`. -/
theorem pr_le_add_off {α : Type} [Fintype α] {A B : α → Prop} {b : ℝ} (h : Pouw.pr (fun x => A x ∧ ¬ B x) ≤ b) :
    Pouw.pr A ≤ b + Pouw.pr B :=
  (Pouw.pr_mono fun x hx => (em (B x)).elim Or.inr fun hb => Or.inl ⟨hx, hb⟩).trans
    ((Pouw.pr_or_le (fun x => A x ∧ ¬ B x) B).trans (add_le_add h le_rfl))

/-- **The network certifier's per-window bound at its compiled component**
(`Verity.Subprotocol.NetworkWindowLe`, unfolded, at the component's acceptance), from S2–S6, the ρ gap and S8's two
premises, with S1, S7 and S8 proved here. At each window and each value of the other coins, an accepted run of the
network audit under S5's guard `G` with window `i` not good maps, through the live coin map (S4), to an accepting run
of the sampled verifier (S2, S3, S5) at whose every extractor coin the extraction fails the leak relation: an extracted
witness would make the realization's witness one (S6), and with the component's acceptance, window `i` good (S8). -/
theorem windowLe (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} {G : ((i : I) → Ω i) → Prop}
    (h2 : PeriodRunsV2 W s) (h3 : V2AcceptMeans W) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G)
    (h6 : NetworkCouples R s W) (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s)
    (hcov : RecordCovers W s) :
    ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      ComponentAccepts .onsite .network (s (Function.update ξ (r.own .network) x)) ∧
        G (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt (Function.update ξ (r.own .network) x) i)
          (R.wit (Function.update ξ (r.own .network) x) i) (R.obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V (Rlt R) (E D decode) (R.stmt ξ i) (R.prover ξ i) := by
  intro ξ i
  refine pr_le_of_coins (W.coins ξ i) (h4 ξ i) fun x ⟨hca, hg, hbad⟩ c =>
    ⟨h5 ξ i x hg (h3 _ i (h2 _ i hca)), fun hx => ?_⟩
  have hex : ∃ w, Rlt R (R.stmt (Function.update ξ (r.own .network) x) i) w := by
    rw [h6.stmt]
    exact exists_of_ext hx
  exact hbad (windowGood_of_accepts W h2 h6 hρ hreg hcov hca (h6.wit _ i hex))

/-- **`windowLe` with S8's coverage off an event `Brk`** (`RecordCoversOff`): the bound gains `Brk`'s probability over
the network audit's own coin. Under B, `Brk` is the forger's event (`CertForged`, `recordCoversOff_of_certified`). -/
theorem windowLeOff (W : WindowRuns R) {s : ((i : I) → Ω i) → Sent} {G : ((i : I) → Ω i) → Prop}
    (h2 : PeriodRunsV2 W s) (h3 : V2AcceptMeans W) (h4 : WindowCoinsLive W) (h5 : SessionIsSampledRun W G)
    (h6 : NetworkCouples R s W) (hρ : WindowValuesRegistered R s) (hreg : WindowRegistersAs W s)
    {Brk : ((i : I) → Ω i) → Prop} (hcov : RecordCoversOff W s Brk) :
    ∀ ξ i, Pouw.pr (fun x : Ω (r.own .network) =>
      ComponentAccepts .onsite .network (s (Function.update ξ (r.own .network) x)) ∧
        G (Function.update ξ (r.own .network) x) ∧
        ¬ WindowGood H D L Kₒ S lay k δ reg Dj (R.sw i) (R.ρ i) R.tops (R.stmt (Function.update ξ (r.own .network) x) i)
          (R.wit (Function.update ξ (r.own .network) x) i) (R.obs (Function.update ξ (r.own .network) x) i)) ≤
    PrBad H D.V (Rlt R) (E D decode) (R.stmt ξ i) (R.prover ξ i) +
      Pouw.pr (fun x : Ω (r.own .network) => Brk (Function.update ξ (r.own .network) x)) := by
  intro ξ i
  refine pr_le_add_off ?_
  refine pr_le_of_coins (W.coins ξ i) (h4 ξ i) fun x ⟨⟨hca, hg, hbad⟩, hnb⟩ c =>
    ⟨h5 ξ i x hg (h3 _ i (h2 _ i hca)), fun hx => ?_⟩
  have hex : ∃ w, Rlt R (R.stmt (Function.update ξ (r.own .network) x) i) w := by
    rw [h6.stmt]
    exact exists_of_ext hx
  exact hbad (windowGood_of_covers W h2 h6 hρ hreg hca (hcov _ i hca hnb) (h6.wit _ i hex))

end Verity.Network
