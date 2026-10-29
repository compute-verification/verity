import NetworkCheck.Period
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The network period check's bound: what its `K` counts in each form (`NetworkCheck.Period`)

On acceptance, `NetworkCheck.Period.check` publishes `K`, the sum of a ceiling of `log2` of each factor of
`NetTiming.Guarantees.PeriodLeakAccepted`'s count at the number the check reads for it: `terms`'s in the clock form,
`rowsTerms`'s in the rows form. A sum of ceilings is at least the ceiling of the sum, so `K` is at least `⌈log2⌉` of
the count, not equal to it; `ClockKBoundsCount` and `RowsKBoundsCount` say the count at the check's numbers is at most
`2 ^ K`. `Proofs.Properties.NetworkPeriod` proves both.

In the rows form the entry bits the tied audits' incorrect proof blocks can set are charged once a period, as the
capacity term `κ_t`: `RowsKBoundsCount` multiplies the windows' factors by `NetTiming.Guarantees.PooledLeakAccepted`'s
tied factor at the check's pool (`tiedFactor`), and `TiedFMeets` says the `F_t` the check computes meets the pooled
bound's premise `ln (1/δ_t) ≤ q_t · F_t`.

The count is the statement's, factor by factor, over the network certifier's `m` connections and the period's `n`
windows: `#Ss · ((∏_c (cap_c + 1)) · ∏_i (2^#(free inputs) · #faultySets · 2^β · Dj^k))`. A reader of `K` (the
communication audit's bridge to `PeriodLeakAccepted`) instantiates it at the run's caps and the windows' facts, and
supplies each of the statement's quantities as at most the check's number:

- in the clock form the windows are the compute audits: `#Ss ≤ #Σ_sync ^ (link-windows)` (each link-window under one
  sync of Σ_sync), an audit's free inputs at most its layout's (`free`), its faulty sets at most the sets of at most `b`
  of its `N` units, `β` at most `b · bits` with `bits` the most one of its Program's units commits (`faulty_bits`), and
  `Dj^k` at most `jitter ^ steps`;
- in the rows form the windows are the link-windows' scans: `#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes'
  lengths), a window's free inputs at most its scan's Program's and its tied audits' (`free`: the egress renders the
  outputs of the audits its scan's `prog` is tied to, so the bits the prover chose of their inputs are the window's),
  each window's faulty sets at most the sets of at most `b` of its `N` units, `β` at most `b · max(b_ing, b_eg)`
  (`faulty_bits`) and, when a tied audit is sampled in part with a bound of at least one incorrect proof block, every
  slot entry bit (`faulty_entries`, `E_w`), and `Dj^k` at most `jitter ^ steps`.

The rows form's terms charge a tied audit (one a scan's `prog` is an output of) its free inputs, and its incorrect
proof blocks only as the slot entries they can set, since the scan reads the audit only through its `prog` and its
state is the scan's own. The check accepts a tied audit at any sample; the consistency check of one is `RowsProgWired`
and `RowsStateTied`. `RowsProgWired` says it accepts no link-window whose scan's `prog` is not a compute audit's slot
entries, a lane's tree the entries port of its chunk of a `SlotEntry_v1` or `SelectEntry_v1` template of that audit's
own Program. `RowsStateTied` says it accepts no link-window whose scan's state is not the `SCAN_STATE` of an audit its
`prog` is wired to, so a state that skips or alters one of the scan's transitions is no Program's input. Each
says its conclusion (`RowsProgWiredAt`, `RowsStateTiedAt`) of every period `check` accepts, and so is, by definition,
its property of any period check at `check` (`RowsProgWiredOf`, `RowsStateTiedOf`), which the nontrivial lemmas state
false at `acceptAllRows`, a check that accepts every period. Each guarantee writes its acceptance of `check` out rather
than being its `…Of check`: the shape rule reads the acceptance off the statement's own binders.
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period

/-- **The clock form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, `n` compute audits `a` and their facts `w`, the jitter bound `jitter`, `#Σ_sync` `syncs` and `windows`
link-windows: `#Ss` at `syncs ^ windows` (`schedules`), `∏_c (cap_c + 1)` at `∏_j (caps j + 1)` (`cells`), and audit
`i`'s `2^#(free inputs)` at `2 ^ (a i).free`, `#faultySets` at `atMost N b`, `2^β` at `2 ^ (b · (a i).bits)` (each of
its at most `b` wrong units commits at most its Program's widest unit's bits) and `Dj^k` at `jitter ^ (a i).steps`,
their product is at most `2 ^ K`. -/
def ClockKBoundsCount : Prop :=
  ∀ (jitter syncs windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (a : Fin n → Audit) (w : Fin n → Facts),
    syncs ^ windows * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (a i).free * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * (a i).bits) * jitter ^ (a i).steps)) ≤
      2 ^ (terms (Array.ofFn a) (Array.ofFn w) (Array.ofFn caps) jitter syncs windows).K

/-- `PooledLeakAccepted`'s tied factor at the check's pool (`Pool`): `min((N_t + 1)^F_t · 2^(R · F_t), 2^span)`, or
`2^span` with none, `span` the link-windows' entry bits their tied audits' incorrect proof blocks can set. -/
def tiedFactor (pool : Option Pool) (span : ℕ) : ℕ := match pool with
  | some p => min ((p.blocks + 1) ^ p.f * 2 ^ (p.reach * p.f)) (2 ^ span)
  | none => 2 ^ span

/-- **The check's `F_t` meets the tied audits' pooled bound's premise.** At every `δ = num / den` and detection
probability `q = k / n` the check computes `F_t` at (`tiedF`), `ln (1/δ) ≤ q · F_t`, so the pooled bound's escape
`(1 - q)^F_t` is at most `δ` (sampled proofs' `PooledIncorrectBlocksLe` at `F_t`). -/
def TiedFMeets : Prop :=
  ∀ (δ : Core.Ratio) (k n F : ℕ), tiedF δ k n = some F →
    Real.log (1 / ((δ.num : ℝ) / δ.den)) ≤ (k : ℝ) / n * F

/-- **The rows form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, the facts `w` of `n` link-windows' scans, their tied audits' free inputs `tied` (`tiedAudits`) and the
slot entry bits their tied audits' incorrect proof blocks can set `faulty` (`faultyEntries`), under one scan's terms
`scan`, the jitter bound `jitter`, the charged audits' facts `charged` and the tied audits' pool `pool` (`tiedPool`):
`#Ss` and `∏_c (cap_c + 1)` each at `∏_j (caps j + 1)` (`schedules` and `cells`), window `i`'s `2^#(free inputs)` at
`2 ^ (scan.free + tied i)`, `#faultySets` at `atMost N b`, `2^β` at `2 ^ (b · max(b_ing, b_eg))` and `Dj^k` at
`jitter ^ scan.steps`, and once a period `PooledLeakAccepted`'s tied factor at the pool and `Σ_i faulty i`
(`tiedFactor`), their product is at most `2 ^ K`. -/
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied faulty : Fin n → ℕ) (charged : Array Facts) (pool : Option Pool),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (scan.free + tied i) * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps)) * tiedFactor pool (∑ i, faulty i) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows
        charged pool).K

/-- What `RowsProgWired` says of a period its check accepted: in the rows form, each link-window's scan's registration
is wired to the audits' slot entries (`wired`). -/
def RowsProgWiredAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → wired p.audits s c.audits reg = true

/-- `RowsProgWired` of the period check `chk`: `RowsProgWiredAt` of every period `chk` accepts. -/
def RowsProgWiredOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsProgWiredAt p c

/-- **The rows form accepts no link-window whose scan's `prog` is not a compute audit's slot entries.** If the check
accepts a period in the rows form, each link-window's scan's registration has, for each lane `j`, its `scan/prog<j>`
entry at the root and domain of one audit's entries port of chunk `j`, `<descriptor id>/c<j>/o0` of `T / every` leaves,
of a template node of that audit's own Program calling `SlotEntry_v1` or `SelectEntry_v1` (whose select and collector
end in `SlotEntry_v1`), a Definition the check pins by its closure digest (`entryPins`, so a template under either name
with another body is refused), with one member a slot of the window (`wired`), so chunk `j` is lane `j`'s slots and each
slot's entry, read bit included, is one its Program computed. -/
def RowsProgWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsProgWiredAt p c

/-- What `RowsStateTied` says of a period its check accepted: in the rows form, each link-window's scan's state
entries are at one root and domain, and every audit wired to it has its own there (`stateTied`). -/
def RowsStateTiedAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → stateTied p.audits s c.audits reg = true

/-- `RowsStateTied` of the period check `chk`: `RowsStateTiedAt` of every period `chk` accepts. -/
def RowsStateTiedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsStateTiedAt p c

/-- **The rows form accepts no link-window whose scan's state is not its wired audit's `SCAN_STATE`.** If the check
accepts a period in the rows form, each link-window's scan's registration has a state entry (`stateEntries`: a `roots`
port `scan/state` or a `registered` value `verity/certifier/scan-state/v1`), every one at one root and domain, and an
audit whose slot entries are the scan's `prog` (`wired`) has one, every one at that root and domain (`stateTied`). The
scan's units read their state at that root and register the state their transition computes there, so the state a
slot's Program reads is the scan's, and a `SCAN_STATE` that skips or alters a transition, at another root or domain, is
refused. -/
def RowsStateTied : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsStateTiedAt p c

/-- A period check that accepts every period, publishing the rows form's terms of no link-window: what
`RowsProgWiredOf` and `RowsStateTiedOf` must fail at for `RowsProgWired` and `RowsStateTied` (each its `…Of` at
`check`) to say something. -/
def acceptAllRows (p : Params) (_ : Commitment) (_ : Evidence) : Core.Outcome Public Profile :=
  .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] p.cells p.jitter p.windows) { link := p.link, windows := p.windows }

end Properties.NetworkPeriod
