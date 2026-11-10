import NetworkCheck.Period
import Mathlib.Algebra.BigOperators.Fin

/-!
# The network period check's bound: what its `K` counts in each form (`NetworkCheck.Period`)

On acceptance, `NetworkCheck.Period.check` publishes `K`, the sum of a ceiling of `log2` of each factor of
`NetTiming.Guarantees.PeriodLeakAccepted`'s count at the number the check reads for it: `terms`'s in the clock form,
`rowsTerms`'s in the rows form. A sum of ceilings is at least the ceiling of the sum, so `K` is at least `⌈log2⌉` of
the count, not equal to it; `ClockKBoundsCount` and `RowsKBoundsCount` say the count at the check's numbers is at most
`2 ^ K`. `Proofs.Properties.NetworkPeriod` proves both.

The count is the statement's, factor by factor, over the network certifier's `m` connections and the period's `n`
windows: `#Ss · ((∏_c (cap_c + 1)) · ∏_i (2^#(free inputs) · #faultySets · 2^β · Dj^k))`. A reader of `K` (the
communication audit's bridge to `PeriodLeakAccepted`) instantiates it at the run's caps and the windows' facts, and
supplies each of the statement's quantities as at most the check's number:

- in the clock form the windows are the compute audits: `#Ss ≤ #Σ_sync ^ (link-windows)` (each link-window under one
  sync of Σ_sync), an audit's free inputs at most its layout's (`free`), its faulty sets at most the sets of at most `b`
  of its `N` units, `β` at most `b · bits` with `bits` the most one of its Program's units commits (`faulty_bits`), and
  `Dj^k` at most `jitter ^ steps`;
- in the rows form the windows are the link-windows' scans: `#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes'
  lengths), each window's faulty sets at most the sets of at most `b` of its `N` units, `β` at most
  `b · max(b_ing, b_eg)` (`faulty_bits`), `Dj^k` at most `jitter ^ steps`, and what the prover chose of the window's
  inputs by its scan's draw:
  - the scan sampled in part: the window's free inputs at most its scan's Program's and its tied audits' (`free`: the
    egress renders the outputs of the audits its scan's `prog` is tied to, so the bits the prover chose of their inputs
    are the window's) and, when a tied audit is sampled in part with a bound of at least one incorrect proof block,
    every slot entry bit (`faulty_entries`, `E_w`);
  - every unit of the scan proved (`Facts.whole`): the scan's Program's free inputs and its slot entries' bits once
    (`entries`, `E_w`), in place of its tied audits' free inputs and `faulty_entries` together. The reader supplies
    that such a window's count is at most `2 ^ E_w` times its scan's own factors; no property here states it.

The rows form's terms charge a tied audit (one a scan's `prog` is an output of) its free inputs, and its incorrect
proof blocks only as the slot entries they can set, since the scan reads the audit only through its `prog` and its
state is the scan's own; where every unit of the scan is proved they charge the scan's slot entries once instead, with
no minimum of the two charges. The check accepts a tied audit at any sample; the consistency check of one is
`RowsProgWired`,
`RowsStateTied` and `RowsCircuitHeld`. `RowsProgWired` says it accepts no phase whose scan's `prog` is not a
compute audit's slot entries, a lane's tree the entries port of its chunk of a `SlotEntry_v1` or `SelectEntry_v1`
template of that audit's own Program. `RowsStateTied` says it accepts no phase whose scan's state is not the
`SCAN_STATE` of an audit its `prog` is wired to, read where the scan's units chain it from row 0, the empty state, so a
state that skips or alters one of the scan's transitions, or starts from a state the prover chose, is no Program's
input. Each says its conclusion (`RowsProgWiredAt`, `RowsStateTiedAt`) of every period `check` accepts, and so is, by definition,
its property of any period check at `check` (`RowsProgWiredOf`, `RowsStateTiedOf`), which the nontrivial lemmas state
false at `acceptAllRows`, a check that accepts every period. Each guarantee writes its acceptance of `check` out rather
than being its `…Of check`: the shape rule reads the acceptance off the statement's own binders.

`RowsCircuitHeld` says the check accepts no phase whose tied audit's registration names a Program other than the class
circuit, the parameters' `circuit`: the one Program the workload's class registers, fixed for the period, so the prover
chooses no phase's circuit. It states the registrations alone: that the descriptor whose input gates a
tied audit's term counts is the Program its registration names is the audit check's (`auditCheck`, which compares
them by matching, `isStr`) and no property here states. Its witnesses are an empty rows-form period
`check` accepts, beside a tied audit registered under the class circuit, and an explicit period whose tied audit is
registered under another (`RowsCircuitHeldAt` false), which a check that accepts everything accepts.

`RowsStateAnchored` says the check accepts no phase whose tied audit's circuit record reads the scan's state other than
at its slots' anchors: lane `l`'s slot at boundary `b`, unit `l · nb + b`, reads row `b · every`, the state before its
anchor's bucket, so with `RowsStateTied` a slot's Program reads the state the scan reached at its anchor. Its witnesses
are an empty rows-form period `check` accepts, beside a tied audit whose record reads row `b` at unit `l · 2 + b`
(`anchorAudit 0`), and an explicit period whose record reads from base 3 (`RowsStateAnchoredAt` false).

`RowsStateHeld` says the check's term of a link-window whose scan is sampled in part counts a tied audit's reads of the
scan's state as no free inputs only where that audit's `SCAN_STATE` is the scan's state, at its registration's root and
domain, in every phase the audit is tied to (`stateHeldIn`), and else counts its whole free inputs: the term never
subtracts what the state tie doesn't fix. The term of a link-window whose scan has every unit proved counts no tied
audit's free inputs and charges the scan's slot entries. Its witnesses are an empty rows-form period `check` accepts,
beside an audit whose `SCAN_STATE` is its one phase's scan's state and is charged its `stateFree`, and an explicit
period and public, of a scan sampled in part, charging `stateFree` for an audit whose `SCAN_STATE` is at another root
(`RowsStateHeldAt` false).
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

/-- **The rows form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, the facts `w` of `n` link-windows' scans, their tied audits' free inputs `tied` (`tiedAudits`) and the
slot entry bits their tied audits' wrong units can set `faulty` (`faultyEntries`), under one scan's terms `scan`, the
jitter bound `jitter` and the charged audits' facts `charged`: `#Ss` and `∏_c (cap_c + 1)` each at `∏_j (caps j + 1)`
(`schedules` and `cells`), and window `i`'s `2^#(free inputs)`, `#faultySets` at `atMost N b`, `2^β` at
`2 ^ (b · max(b_ing, b_eg))` times its entries' factor and `Dj^k` at `jitter ^ scan.steps`, their product is at most
`2 ^ K`. Window `i`'s free inputs and entries' factor are by its scan's draw: every unit of the scan proved
(`Facts.whole`), `2 ^ (scan.free + scan.entries)`, the scan's slot entries once, and no factor more, whatever `tied i`
and `faulty i`; the scan sampled in part, `2 ^ (scan.free + tied i)` and `2 ^ faulty i`. This is the arithmetic alone:
that a period's count is this product at these numbers is the bound's reader's to supply. -/
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts)
    (tied faulty : Fin n → ℕ) (charged : Array Facts),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i,
        (2 ^ (scan.free + if (w i).whole then scan.entries else tied i) * atMost (w i).units (w i).wrong *
          2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps *
          2 ^ (if (w i).whole then 0 else faulty i))) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows
        charged).K

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

/-- What `RowsStateTied` says of a period its check accepted: in the rows form, each link-window's scan's registration
states its units' state reads as the scan's transitions chain them, its state entries are at one root and domain, an
audit wired to it has its own there (`stateTied`), and it opens the state's row 0 to the empty state (`stateEmpty`). -/
def RowsStateTiedAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg →
      stateTied p.audits s c.audits reg = true ∧ stateEmpty s reg = true

/-- `RowsStateTied` of the period check `chk`: `RowsStateTiedAt` of every period `chk` accepts. -/
def RowsStateTiedOf (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    chk p c e = .accept pub prof → RowsStateTiedAt p c

/-- **The rows form accepts no link-window whose scan's state is not its wired audit's `SCAN_STATE`.** If the check
accepts a period in the rows form, each link-window's scan's registration has a state entry (`stateEntries`: a `roots`
port `scan/state` or a `registered` value `verity/certifier/scan-state/v1`), every one at one root and domain, and an
audit whose slot entries are the scan's `prog` (`wired`) has one (`auditStateEntries`: those, or its `window`'s
`activations.values` entry of that value), every one at that root and domain; and its `window`'s
`reads` of `scan/state` state ingress unit `t`'s `sin` at leaf `t`, `t` from 1, and its `o1` at leaf `t + 1`, `t` below
`T − 1`, one run a port (`stateRead`), the reads the scan's verifier of record holds each unit's ports to; and its
`window`'s `empty` of `scan/state` opens leaf 0 of that root and domain to `W` zero u16 words, `W` at least 1, under a
salt of 192 bytes through one sibling a level of the `T` leaves (`stateEmpty`). So the scan's units read their state
at that root where the unit before registered it, from row 0, which no unit computes and the Programs anchored at
bucket 0 read, opened to the empty state; the state a slot's Program reads is the scan's, and a `SCAN_STATE` that
skips or alters a transition, at another root or domain, a registration whose units read their state off the chain
(a `sin` at leaf `t + 1`, an `o1` at leaf `t`), or one that opens no leaf 0 to the empty state, is refused. -/
def RowsStateTied : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsStateTiedAt p c

/-- What `RowsCircuitHeld` says of a period its check accepted: in the rows form, each phase's tied compute audits are
registered under the parameters' class circuit (`held`). -/
def RowsCircuitHeldAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg → held p.circuit c.audits reg = true

/-- **The rows form accepts no phase whose tied compute audit's window circuit is not the class circuit.** If the check
accepts a period in the rows form, every audit a phase's scan's `prog` is an output of (`tiedAudits`) has a
registration whose `program` is the parameters' `circuit`, the SHA-512 of the Program of the one window circuit the
workload's class registers (`held`). A window circuit's own digest binds its window's run; its Program is one for the
period, so no phase's tied audit is registered under a Program the prover chose for that phase. That its term
counts the free inputs of that Program's descriptor is the audit check's (`auditCheck`), not stated here. -/
def RowsCircuitHeld : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsCircuitHeldAt p c

/-- Registered read `x` of the scan's state, in a record of wiring `c` and parameters `params`, reads it at its slots'
anchors under schedule `s`: it is of an input port of its reader template, uncut and of `W` words, from a parameter of
`T · W` leaves, the scan's `T` state rows, over the units `l · nb + b` for each lane `l` and boundary `b` below
`nb` = T / `every` (lane-major, as `wired` reads chunk `l`'s leaf `b`), each reading its port's one row from the
parameter's leaf `b · every · W`: row `b · every`, the scan's state before its anchor's bucket. -/
def AnchoredRead (s : Schedule) (c : NetworkCheck.Circuit.Wiring) (params : List (String × ℕ))
    (x : NetworkCheck.Circuit.Read) : Prop :=
  (NetworkCheck.Circuit.portOf c x.pat.reader x.pat.port false).isSome = true ∧
    ∀ (W k : ℕ), NetworkCheck.Circuit.portOf c x.pat.reader x.pat.port false = some (W, k) →
      k = 1 ∧ params.lookup stateValue = some (s.window * W) ∧ x.pat.units = s.lanes * (s.window / s.every) ∧
      ∀ l < s.lanes, ∀ b < s.window / s.every,
        x.pat.readerAt (l * (s.window / s.every) + b) = (l * (s.window / s.every) + b, 0) ∧
        x.pat.writerAt (l * (s.window / s.every) + b) = ((b * s.every * W : ℕ) : ℤ)

/-- Circuit record `r` reads the scan's state at its slots' anchors under schedule `s`: no table of it is over the
scan's state (`stateValue`), and each of its registered reads of it is the anchors' (`AnchoredRead`). -/
def AnchoredRecord (s : Schedule) (r : NetworkCheck.Circuit.Stated) : Prop :=
  (∀ t ∈ r.registered.tables, t.param ≠ stateValue) ∧
    ∀ x ∈ r.registered.reads, x.param = stateValue → AnchoredRead s r.wiring r.registered.params x

/-- What `RowsStateAnchored` says of a period its check accepted: in the rows form, each audit tied to a
link-window's scan (`tiedAudits`) with a circuit record reads the scan's state at its slots' anchors
(`AnchoredRecord`). -/
def RowsStateAnchoredAt (p : Params) (c : Commitment) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg →
      ∀ j ∈ tiedAudits c.audits reg, ∀ (a : Audit) (r : NetworkCheck.Circuit.Stated), p.audits[j]? = some a →
        a.circuit = some r → AnchoredRecord s r

/-- **The rows form accepts no link-window whose tied audit's circuit record reads the scan's state off its slots'
anchors.** If the check accepts a period in the rows form, each audit a link-window's scan's `prog` is an output of
(`tiedAudits`) whose circuit record reads `verity/certifier/scan-state/v1` reads it, unit `l · nb + b` (lane `l`'s
slot at boundary `b`) at row `b · every` of the scan's `T` state rows, whole and from base 0, and declares no table
over it (`anchoredAt`). With `RowsStateTied`, which holds those rows to the scan's chain from the empty state, the
state a slot's Program reads is the one the scan reached at its anchor, never a later one past the requests the scan
admitted there. -/
def RowsStateAnchored : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsStateAnchoredAt p c

/-- What `RowsStateHeld` says of a period its check accepted with public `pub`: in the rows form, each link-window's
term, by its scan's facts, either charges the scan's slot entries (`Scan.entries`) and as free inputs its scan's
alone, where every unit of the scan is proved (`Facts.whole`); or, where the scan is sampled in part, has as free
inputs its scan's and, for each audit tied to it (`tiedAudits`), that audit's free inputs with its reads of the scan's
state held (`Audit.stateFree`) where its `SCAN_STATE` is the scan's state in every link-window tied to it
(`stateHeldIn`), else its whole free inputs (`Audit.free`). -/
def RowsStateHeldAt (p : Params) (c : Commitment) (pub : Public) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json), c.windows[i]? = some reg →
      ∃ t ∈ pub.units[i]?,
        (t.facts.whole = true → t.free = scan.free ∧ t.entries = scan.entries) ∧
        (t.facts.whole = false → t.free = scan.free + (tiedAudits c.audits reg).foldl (fun n j => n +
          (p.audits[j]?.map fun a => if stateHeldIn c.audits c.windows j then a.stateFree else a.free).getD 0) 0)

/-- **The rows form charges a tied audit its reads of the scan's state only where they are of the scan's own state.**
If the check accepts a period in the rows form, its public's term of each link-window whose scan is sampled in part
counts as free inputs its scan's
and, for each audit its scan's `prog` is an output of (`tiedAudits`), that audit's: with its reads of
`verity/certifier/scan-state/v1` held (`Audit.stateFree`) only where its registration has a state entry
(`auditStateEntries`: of its `roots` or `registered`, or its `window`'s `activations.values` entry of that value) and, in
every link-window it is tied to, each of its state entries is at the root and domain of that link-window's scan's state
(`stateHeldIn`), the pin taken from the scan's registration, never from the parameters; else all its free inputs
(`Audit.free`). So the term leaves out no input the prover chose of an audit whose state reads the scan's chain does
not fix (`RowsStateTied` holds the scan's state to that chain from the empty state). That `stateFree` counts the audit's
circuit record with the scan's state as a pinned parameter, and `free` without it, is the parameters' reading
(`auditOf`), not stated here. The term of a link-window whose scan has every unit proved (`Facts.whole`) counts no
tied audit's free inputs, held or not: it charges the scan's slot entries (`Scan.entries`) in their place, and as free
inputs the scan's alone. That those entries bound what such a link-window's egress can carry of its tied audits'
inputs is no statement here. -/
def RowsStateHeld : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsStateHeldAt p c pub

/-- A period check that accepts every period, publishing the rows form's terms of no link-window: what
`RowsProgWiredOf` and `RowsStateTiedOf` must fail at for `RowsProgWired` and `RowsStateTied` (each its `…Of` at
`check`) to say something. -/
def acceptAllRows (p : Params) (_ : Commitment) (_ : Evidence) : Core.Outcome Public Profile :=
  .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] p.cells p.jitter p.windows) { link := p.link, windows := p.windows }

/-- A phase's window record whose `prog` is at the root `r` and the domain `d`. -/
def tiedWindow : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj [("port", "scan/prog0"), ("root", "r"), ("domain", "d")]])]

/-- A compute audit whose Program has one `SlotEntry_v1` template of one member, a pinned Definition (its closure
digest `slotEntryPins`' first), and the schedule of one slot a window (one lane, T = 1, `every` 1). -/
def slotAudit : Audit := ⟨0, 0, "", 1, 0, #[("SlotEntry_v1{KO=1}", 1, slotEntryPins[0]!)], none, none, 0⟩

def slotSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1⟩

/-- The slot audit's registration under the Program `d`: its entries port at the root `r` and the domain `d`, so
`tiedWindow`'s scan's `prog` is its output. -/
def heldReg (d : String) : Lean.Json :=
  Lean.Json.mkObj [("program", .str d), ("roots", Lean.Json.arr #[Lean.Json.mkObj
    [("port", "SlotEntry_v1{KO=1}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]])]

/-- Two lanes, T = 2, `every` 1: two boundaries, four slots. -/
def anchorSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 2⟩

/-- A circuit record of one template `S` of 4 units, its input port `state` of 3 words, whose one registered read is
`read`, over the scan's state as a parameter of 6 leaves (the two state rows of 3 words). -/
def anchorRecord (read : NetworkCheck.Circuit.Read) : NetworkCheck.Circuit.Stated :=
  { templates := [("S", 4)], cap := 0,
    wiring := { templates := [("S", { units := 4, rows := [("state", ⟨3, 1⟩)], outs := [("o0", 1)] })], cuts := [],
                wires := [] },
    registered := { params := [(stateValue, 6)], reads := [read] } }

/-- `S`'s read of the scan's state from `base`: unit `l · 2 + b` at leaf `base + 3b`, so row `b` at base 0. -/
def stateReadAt (base : ℤ) : NetworkCheck.Circuit.Read :=
  { param := stateValue, pat := { reader := "S", port := "state", writer := stateValue, out := "", start := 0,
                                  radix := [2, 2], base, step := [0, 3], cut0 := 0, cuts := 1 } }

/-- The slot audit with the circuit record whose read of the scan's state is from `base`. -/
def anchorAudit (base : ℤ) : Audit := { slotAudit with circuit := some (anchorRecord (stateReadAt base)) }

/-- The slot audit's registration with its `SCAN_STATE` registered at the root `root` and the domain `e`, beside its
entries port at `r` and `d`. -/
def stateReg (root : String) : Lean.Json :=
  Lean.Json.mkObj [
    ("registered", Lean.Json.arr #[Lean.Json.mkObj
      [("value", "verity/certifier/scan-state/v1"), ("root", root), ("domain", "e")]]),
    ("roots", Lean.Json.arr #[Lean.Json.mkObj
      [("port", "SlotEntry_v1{KO=1}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]])]

/-- A run of the scan's state reads: units `lo` to `hi − 1`, unit `lo + i` at leaf `l + i` (`isRun`). -/
def stateRun (l lo hi : ℕ) : Lean.Json :=
  Lean.Json.mkObj [("leaf", Lean.toJson l), ("units", Lean.Json.arr #[Lean.toJson lo, Lean.toJson hi])]

/-- A link-window whose scan's `prog` is at the slot audit's entries port's root and domain, whose state, the tree its
units read and register their states in, is at the root `s` and the domain `e`, and whose registration states its
units' `sin` reads as the run `sin` and their `o1` as `o1`. -/
def stateWindowOf (sin o1 : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [
    ("roots", Lean.Json.arr #[
      Lean.Json.mkObj [("port", "scan/prog0"), ("root", "r"), ("domain", "d")],
      Lean.Json.mkObj [("port", "scan/state"), ("root", "s"), ("domain", "e")]]),
    ("window", Lean.Json.mkObj [("reads", Lean.Json.mkObj [("scan/state", Lean.Json.mkObj [("o1", o1), ("sin", sin)])])])]

/-- That link-window at T = 1, its one unit reading no state and registering none: both runs empty. -/
def stateWindow : Lean.Json := stateWindowOf (stateRun 1 1 1) (stateRun 1 0 0)

/-- The slot audit with 9 free inputs, 4 of them with its reads of the scan's state held. -/
def heldAudit : Audit := { slotAudit with free := 9, stateFree := 4 }

/-- A rows-form period of the held audit and one phase, under the slot audit's schedule. -/
def heldParams : Params :=
  ⟨#[heldAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none, .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩

/-- A public whose one link-window's term, of a scan sampled at one of its two units, counts 4 free inputs: the held
audit's `stateFree`. -/
def heldPublic : Public :=
  { audits := #[], cells := 0, schedules := 0, windows := 1,
    units := #[{ facts := { units := 2, drawn := 1, delta := ⟨0, 1⟩, wrong := 0, draw := .subset 1 }, free := 4,
                 faultySets := 0, faultyBits := 0, jitter := 0 }] }

end Properties.NetworkPeriod
