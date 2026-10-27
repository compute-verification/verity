import Properties.NetworkScan
import Definitions.NetworkCertifier.TapRows
import Definitions.NetworkCertifier.Ingress
import Definitions.NetworkCertifier.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The server's ingress over a period of the rows form: the content the warden lets in (`NetworkCheck.Period`)

In the rows form the network certifier runs as an active warden on the server's uplink, and each link-window's scan
(`ingress_scan`, one one-stage registration a link-window) reads the window's ingress rows: step `t`'s row is bucket
`t`'s `row` parameter, `S = lanes · slots` cells of `B` bytes, an absent cell `B` zero bytes. Every bucket's unit
returns an `ok` bit (leaf 0 of its `u16`), which the verifier opens and requires to be 1, and its parse state (leaves
`16 …` of its return), which the next bucket reads as its `sin` parameter. The scan's `ok` holds only on a row whose
cells are the next frames of their slots' id requests, so an accepted window's ingress is a sequence of id requests
of the scan class's shape. `PerServerIngress` states what that gives about the server's ingress over the period, for
every prover:

* **soundness**: over the period's live coins, the check of record accepts while some window's ingress, as the network
  certifier registers it at the scan's ingress gates, is not one the scan's Program accepts (`IngressDeclared`) with
  probability at most `Σ_i (δ + ε_ks + ε_bind + Pr[B_tap])`, each window's errors at its statement;
* **the count**: the secrets an outside sender passes into the server through the warden, whose every window's ingress
  the scan accepts, number at most
  `∏_i Σ_(F faulty at δ) 2^(#ingress(F) · faultyIngressBits + windowBits) · (∏_ℓ (r ℓ + 1))^n · D^n`. Where the
  draw takes every ingress unit, no faulty set holds one, and the content is `log2 #faultySets + windowBits` bits a
  window (`PerServerIngressDrawn`).

`windowBits` is `N · requestBits + T · bucketBits`, at the window's most requests `N = lanes · ⌈T / (D + 1)⌉`.
`requestBits` is a request's ordinal (32: the scan checks only that it is above the last), its head (content-length,
`max_tokens` and `seed` 32 each, `temperature` and `top_p` 24 each, `model` and `user` 406 each), and its prompt ids
(`K0 + (C − 1)·K` of `IDB` bits); `bucketBits` is each lane's block, absent or `d ≤ r` data cells then padding. A faulty
ingress unit's `ok` is the prover's: its bucket's whole row enters, and its forged state lets each lane carry two
requests more (`faultyIngressBits`). At `W63` of `benchmarks/network_traces/scan_zk.py` (T = 32, D = 27) a request is
36,382 bits, a bucket 12, a window 291,440, and a faulty ingress unit 356,592.

**What it reads.** The soundness reads the check of record through `AnswersOfRecord` (an accepted window's answer is
the sampled verifier's acceptance of that window's statement on that window's coins), and each window's statement
through `ScanScope` (the sampled audit's scope at the scan's layout, with the gates the scan checks committed). The
count reads the warden's record (`RecordsCells`), the openings the verifier holds at the scan's gates (`OpensPass`,
`WardenRowsOpen`, `OkOpens`, and `ChainOpens`, which holds the registered-read chain of parse states as equal
openings), and `ScanCountsRequests`, the count of one window's frame streams the scan accepts, a conjecture on the scan
class's Program checked by enumeration at toy classes. Each window's prover may read every other window's coins
(`WindowTape`), so the period's provers may be adaptive.

**The answers.** `rowsWindowCheck` reads each link-window's answer from the evidence (`answered`) and does not run the
sampled verifier itself, so `AnswersOfRecord` is what ties an accepted answer to that window's coins.
`PerServerIngressHonest` reads acceptance through `HonestWindowsAccepted`.
-/

namespace Properties.NetworkCertifier

open NetTiming Claims.Sampling
open NetworkCheck.Period (Schedule Scan Params Evidence Public Profile rowsWindowCheck auditCheck check clog2 outBits
  templateNodes)
open Properties.NetworkScan (gateAt layOf)

/-! ## The scan class's numbers -/

/-- The scan class's public shape: `C` cells a request at most and `idb`-bit ids. -/
structure ScanClass where
  C : ℕ
  idb : ℕ

/-- `K = (B − 9) / 11` ids a cell: the warden-form id request's prompt fields (`[` or `,`, then a u32 in 10 characters)
after the cell's 9-byte header. -/
def idsPerCell (s : Schedule) : ℕ := (s.cellBytes - 9) / 11

/-- The first prompt field's offset in a request's first cell (`io_request.layout(B).start`): the request's HTTP head,
its content-length and its body up to `"prompt":`, the same at every `B` whose first cell holds them. -/
def promptStart : ℕ := 367

/-- `K0 = (B − 9 − 367) / 11` ids in a request's first cell, after its head. -/
def firstCellIds (s : Schedule) : ℕ := (s.cellBytes - 9 - promptStart) / 11

/-- A request's most prompt ids: `K0` in its first cell and `K` in each other of its `C`. -/
def requestIds (s : Schedule) (cls : ScanClass) : ℕ := firstCellIds s + (cls.C - 1) * idsPerCell s

/-- A name's bits (`model`, `user`, which the scan checks only for form): a quote, up to 62 of the 93 printable ASCII
characters but `"` and `\`, a quote and spaces, `⌈log2 Σ_(k ≤ 62) 93^k⌉`. -/
def nameBits : ℕ := 406

/-- A request's head's bits: its content-length (free while the request is unfinished), `max_tokens` and `seed` (u32
each), `temperature` and `top_p` (`%8.6f` in `[0, 10)`, 10^7 values, 24 bits each), and its two names. -/
def headBits : ℕ := 32 + 32 + 32 + 24 + 24 + 2 * nameBits

/-- **A request's bits**: its connection's ordinal (a u32, which the scan checks only above the last), its head, and its
prompt ids. -/
def requestBits (s : Schedule) (cls : ScanClass) : ℕ := 32 + headBits + requestIds s cls * cls.idb

/-- **A bucket's bits**: each lane's block is absent, or `d ≤ r` data cells then padding, `r + 2` cases. -/
def bucketBits (s : Schedule) : ℕ := s.lanes * clog2 (s.slots + 2)

/-- A window's most requests: a slot is held `D + 1` buckets from its admission, `lanes · ⌈T / (D + 1)⌉`. -/
def windowRequests (s : Schedule) : ℕ := s.lanes * ((s.window + s.delay) / (s.delay + 1))

/-- **A window's ingress bits**, every unit correct: its requests' and its buckets'. -/
def windowBits (s : Schedule) (cls : ScanClass) : ℕ := windowRequests s * requestBits s cls + s.window * bucketBits s

/-- The period's ingress bits over `W` windows, every ingress unit drawn. -/
def ingressBits (s : Schedule) (cls : ScanClass) (W : ℕ) : ℕ := W * windowBits s cls

/-- **A faulty ingress unit's bits**: its `ok` is the prover's, so its bucket's whole row (`8 · S · B` bits) enters,
and its forged state lets each lane carry two requests more, the forged state's own and one admission its reset
allows. -/
def faultyIngressBits (s : Schedule) (cls : ScanClass) : ℕ :=
  8 * s.cellBytes * s.allocation + 2 * s.lanes * requestBits s cls

open Classical in
/-- **The faulty set's ingress units**: `F`'s units that hold a gate the scan checks or a gate of its rows. A faulty
unit that holds neither (an egress unit) frees no ingress. -/
noncomputable def faultyIngress {S : ℕ} (P : Partition) (lay : Layout 1 S) (k : ℕ) (ok : List ℕ) (F : Finset ℕ) :
    ℕ :=
  (F.filter fun v => ∃ g ∈ RU.gates (P.rus.getD v []), g ∈ ok ∨ ∃ t < k, g ∈ lay.ingress t).card

/-! ## The gates the scan checks -/

/-- **The scan's checked gates** in its Program `P` at the schedule `s`: each bucket's `ok` gate (leaf 0 of its return:
bucket 0's first-template member, bucket `t ≥ 1`'s ingress-template member `t − 1`), and each link of the parse state,
bucket `t + 1`'s `sin` parameter leaf `j` read from bucket `t`'s state return leaf `16 + j`. -/
def scanChecks (P : Flock.Program.Prog) (s : Schedule) : List ℕ × List (ℕ × ℕ) :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ts := templateNodes P
  let k0 := ts.headD 0
  let k1 := ts.getD 1 0
  let d1 := P.defs[nodes[k1]!.fn]!
  let ret := fun t j =>
    if t = 0 then gateAt P P.rootScope ⟨false, k0, j⟩ else gateAt P P.rootScope ⟨false, k1, (t - 1) * d1.outLeaves + j⟩
  let ns := ((d1.params[1]?).map (·.2.leaves)).getD 0
  ((List.range s.window).map fun t => ret t 0,
    (List.range (s.window - 1)).flatMap fun t => (List.range ns).map fun j =>
      (gateAt P (P.child P.rootScope k1 t) ⟨true, 1, j⟩, ret t (outBits + j)))

/-- A statement's checked gates at the schedule `s`: its descriptor's, none where it doesn't decode. -/
def checksOf (s : Schedule) (x : Stmt) : List ℕ × List (ℕ × ℕ) :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => scanChecks P s
  | .error _ => ([], [])

/-- **The checked gates are committed**: every `ok` gate and every link's written end is a gate `C` returns, and every
link's read end is an input. -/
def ChecksCommitted (C : Prog) (ok : List ℕ) (links : List (ℕ × ℕ)) : Prop :=
  (∀ g ∈ ok, g ∈ C.outs) ∧ ∀ e ∈ links, e.2 ∈ C.outs ∧ C.gates[e.1]? = some .input

/-! ## What the scan accepts -/

/-- **The ingress the scan's Program accepts** over the horizon `k` with a faulty set `RAcc δ` allows: the window's rows
`ins`, with some advice `a` at the inputs besides the ingress and values `free` held only at the faulty set's committed
gates, under which every `ok` gate is 1 and every state link reads what the previous bucket wrote. -/
def ScanDeclared {S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : Layout 1 S) (k : ℕ) (ok : List ℕ)
    (links : List (ℕ × ℕ)) (δ : ℝ) : Set (List (List Bool)) :=
  {ins | ins.length = k ∧ ∃ F ∈ faultySets d P δ, ∃ (a : ℕ → Bool) (free : ℕ → Option Bool),
    (∀ g b, free g = some b → FaultyGate C P F g) ∧
    (∀ e ∈ links, a e.1 = valOver C (lay.inputOf k ins a) free e.2) ∧
    ∀ g ∈ ok, valOver C (lay.inputOf k ins a) free g = true}

/-- **The openings the verifier holds pass** (named assumption): each of `ops` passes under the statement's roots. -/
def OpensPass (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (st : Stmt) (ops : List (Opening Bool)) : Prop :=
  ∀ q ∈ ops, Cm.Opens H st.roots q = true

/-- **The warden registers the window's rows** (named assumption; `trusted-nebius-host`, as `TapOpens`): the window
is `T` rows, and every ingress gate of its steps has an opening in `ops`, whose value is that
bit of the rows `ins`. -/
def WardenRowsOpen (s : Schedule) (st : Stmt) (ops : List (Opening Bool)) (ins : List (List Bool)) : Prop :=
  ins.length = s.window ∧
    ∀ t < s.window, (ins.getD t []).length = ((layOf s st).ingress t).length ∧
      ∀ j g, ((layOf s st).ingress t)[j]? = some g →
        (∃ q ∈ ops, q.pos = g) ∧ tapVal ops g = (ins.getD t []).getD j false

/-- **Every bucket's `ok` opens to 1** (named assumption: the sampled verifier opens each unit's `o0` and requires
1): every `ok` gate has an opening in `ops`, of value 1, and is no constant 0 (as `TapOpens` reads a constant gate's
opening). -/
def OkOpens (s : Schedule) (st : Stmt) (ops : List (Opening Bool)) : Prop :=
  ∀ g ∈ (checksOf s st).1, (∀ b, st.prog.gates[g]? = some (.const b) → b = true) ∧ (∃ q ∈ ops, q.pos = g) ∧
    tapVal ops g = true

/-- **The state chain opens as one value** (named assumption: the registered read of the previous bucket's state,
which the Claims model has no form for): both ends of every state link have an opening in `ops`, of one value, and a
written end that is a constant opens to it (as `TapOpens` reads a constant gate's opening). -/
def ChainOpens (s : Schedule) (st : Stmt) (ops : List (Opening Bool)) : Prop :=
  ∀ e ∈ (checksOf s st).2, (∃ q ∈ ops, q.pos = e.1) ∧ (∃ q ∈ ops, q.pos = e.2) ∧ tapVal ops e.1 = tapVal ops e.2 ∧
    ∀ b, st.prog.gates[e.2]? = some (.const b) → tapVal ops e.2 = b

/-- **The openings `ops` register the window's rows `ins` at the scan's gates**: all four of the above. -/
def IngressOpens (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (s : Schedule) (st : Stmt)
    (ops : List (Opening Bool)) (ins : List (List Bool)) : Prop :=
  OpensPass H Cm st ops ∧ WardenRowsOpen s st ops ins ∧ OkOpens s st ops ∧ ChainOpens s st ops

/-- **The window's registered ingress is one the scan accepts**: the statement `x` parses and has a partition, and every
window the openings `ops st` register at its gates is in `ScanDeclared` at its Program, partition and design. -/
def IngressDeclared {Coin Col T : Type} (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (s : Schedule)
    (ops : Stmt → List (Opening Bool)) (δ : ℝ) (x : Claims.Msg) : Prop :=
  ∃ st P, D.parse x = some st ∧ D.partition st = some P ∧
    ∀ ins, IngressOpens H D.commit s st (ops st) ins →
      ins ∈ ScanDeclared st.prog P st.design (layOf s st) s.window (checksOf s st).1 (checksOf s st).2 δ

/-- **A link-window's scan statement**: in the sampled audit's scope at the scan's layout and horizon (`ScopeAcc`), and
its checked gates committed. `ScanScopeAcc` and the scan's shape (`scanOf`) give it. -/
def ScanScope {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (D : Deploy Coin Col T) (s : Schedule) (x : Claims.Msg) : Prop :=
  ScopeAcc D (layOf s) (fun _ => s.window) x ∧
    ∀ st, D.parse x = some st → ChecksCommitted st.prog (checksOf s st).1 (checksOf s st).2

/-! ## The warden's record -/

/-- A record row's bits as the scan's `row` parameter reads them: its slots in order, a present slot its frame's bits,
an absent one `8B` zero bits. -/
def rowBits {S : ℕ} (s : Schedule) (r : Row S (List Bool)) : List Bool :=
  (List.ofFn r).flatMap fun c => c.getD (List.replicate (8 * s.cellBytes) false)

/-- Window `i`'s ingress in the record `o` of the requests' link: its `T` steps' rows, as bits. -/
def windowIns {K S : ℕ} (s : Schedule) (o : List (Obs 1 K S)) (i : ℕ) : List (List Bool) :=
  (List.range s.window).map fun t => ((o[i * s.window + t]?).map fun y => rowBits s (y.rows 0)).getD []

/-- A row's cells of `8B` bits, in order. -/
def cellsOf (s : Schedule) (bits : List Bool) : List (List Bool) :=
  (List.range (bits.length / (8 * s.cellBytes))).map fun k => (bits.drop (k * (8 * s.cellBytes))).take (8 * s.cellBytes)

/-- A window's frame stream: its rows' cells that aren't padding, in order. -/
def frameStream (s : Schedule) (ins : List (List Bool)) : List (List Bool) :=
  ins.flatMap fun row => (cellsOf s row).filter fun c => c.any id

/-- **The warden's record holds cells** (named assumption): every frame it releases on the requests'
link is one cell, `8B` bits, and not all zero, so padding is the zero cell. -/
def RecordsCells {K S : ℕ} (s : Schedule) (o : List (Obs 1 K S)) : Prop :=
  ∀ y ∈ o, ∀ σ c, y.rows 0 σ = some c → c.length = 8 * s.cellBytes ∧ c.any id = true

/-- **One window's bound**: over the faulty sets `RAcc δ` allows, the window's bits and each of its faulty ingress
units' (`faultyIngressBits`). Where no faulty set holds an ingress unit it is `#faultySets · 2^windowBits`. -/
noncomputable def scanBound (s : Schedule) (cls : ScanClass) (st : Stmt) (P : Partition) (δ : ℝ) : ℕ :=
  ∑ F ∈ faultySets st.design P δ,
    2 ^ (faultyIngress P (layOf s st) s.window (checksOf s st).1 F * faultyIngressBits s cls + windowBits s cls)

/-- **The draw takes every ingress unit**: every unit of the partition `P` that holds a gate the scan checks or a gate
of its rows is in a class its design draws `all` (`allSlots`), so no faulty set holds one. -/
def IngressDrawn (s : Schedule) (st : Stmt) (P : Partition) : Prop :=
  ∀ v < P.rus.length, (∃ g ∈ RU.gates (P.rus.getD v []), g ∈ (checksOf s st).1 ∨ ∃ t < s.window,
    g ∈ (layOf s st).ingress t) → v ∈ allSlots st.design P

/-- **The scan counts requests** (named assumption): the frame streams of the windows the scan's
Program accepts at the statement `st` are at most `scanBound`. A conjecture on the scan's Program, which
`verity/core/protocols/certifier/tests/test_network_certifier_scan_counts.py` tries to falsify against the scan's
reference: each term's field checks at their edges, and every window of rows over a cell alphabet at toy classes, from
the empty state and from a faulty unit's forged states. -/
def ScanCountsRequests (s : Schedule) (cls : ScanClass) (st : Stmt) (P : Partition) (δ : ℝ) : Prop :=
  (frameStream s '' ScanDeclared st.prog P st.design (layOf s st) s.window (checksOf s st).1 (checksOf s st).2 δ).encard ≤
    (scanBound s cls st P δ : ℕ∞)

/-! ## The check of record -/

/-- The outcome accepts. -/
def PeriodAccepts {Pub Prof : Type} (o : Core.Outcome Pub Prof) : Prop := ∃ x pr, o = .accept x pr

/-- Link-window `i`'s registration and evidence are there, and the rows form's window check accepts them. -/
def WindowAnswered (p : Params) (s : Schedule) (sd : ByteArray) (c : NetworkCheck.Period.Commitment) (e : Evidence)
    (i : ℕ) : Prop :=
  ∃ reg x f, c.windows[i]? = some reg ∧ e.windows[i]? = some x ∧ rowsWindowCheck p s sd (p.first + i) reg x = .ok f

/-- The period's coins: the prover's tape and each window's live coins. -/
abbrev PeriodCoins {Coin Col T : Type} (D : Deploy Coin Col T) {n : ℕ} (x : Fin n → Claims.Msg) (Ω : Type) : Type :=
  Ω × ((i : Fin n) → Fin (D.V.k (x i)) → Coin)

/-- Window `i`'s prover's tape: the prover's own and the live coins of every other window. A window's prover has seen
the earlier windows' coins; the tape grants it the later windows' too. -/
abbrev WindowTape {Coin Col T : Type} (D : Deploy Coin Col T) {n : ℕ} (x : Fin n → Claims.Msg) (Ω : Type) (i : Fin n) :
    Type :=
  Ω × ((j : {j : Fin n // j ≠ i}) → Fin (D.V.k (x j.1)) → Coin)

/-- Window `i`'s prover's tape on the period's coins `ω`. -/
def tapeOf {Coin Col T : Type} {D : Deploy Coin Col T} {n : ℕ} {x : Fin n → Claims.Msg} {Ω : Type}
    (ω : PeriodCoins D x Ω) (i : Fin n) : WindowTape D x Ω i :=
  (ω.1, fun j => ω.2 j.1)

/-- **The answers are of record** (named assumption): the evidence `ev ω` is the provers' on the
period's coins `ω`, and where its window `i` is answered (`WindowAnswered`), the sampled verifier's run on that window's
statement `x i`, with window `i`'s prover `P i` at its tape (`tapeOf`) and on that window's coins, accepted: the
answer is the verifier's own. -/
def AnswersOfRecord {Coin Col T : Type} [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T)
    (p : Params) (s : Schedule) (sd : ByteArray) (c : NetworkCheck.Period.Commitment) (x : Fin p.windows → Claims.Msg)
    {Ω : Type} (P : (i : Fin p.windows) → Claims.Prover (WindowTape D x Ω i)) (ev : PeriodCoins D x Ω → Evidence) :
    Prop :=
  ∀ ω i, WindowAnswered p s sd c (ev ω) i.val → (Claims.run H (P i) D.V (x i) (tapeOf ω i) (ω.2 i)).out = .acc

/-- The period's error: each window's `δ + ε_ks + ε_bind + Pr[B_tap]` at its statement and its prover. -/
noncomputable def periodErr {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (δ : ℝ) (decode : List (ℕ × Col) → List T)
    (ops : Stmt → List (Opening Bool)) {n : ℕ} (x : Fin n → Claims.Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : (i : Fin n) → Claims.Prover (WindowTape D x Ω i)) : ℝ :=
  ∑ i, (κ D δ (x i) + Real.sqrt (γ D (x i) * Claims.PrB H D.V (P i) (x i)) +
    (Bext H D.V (E D decode) (WindowTape D x Ω i) (P i) (x i)).Pr H +
    (Btap H D.V (E D decode) D.commit D.parse ops (WindowTape D x Ω i) (P i) (x i)).Pr H)

/-- **`chk`'s acceptance is sound for the period's ingress**: over the period's coins, `chk` accepts the evidence while
some window's registered ingress is not one the scan accepts with probability at most `periodErr`. -/
def PeriodSound (chk : Params → NetworkCheck.Period.Commitment → Evidence → Core.Outcome Public Profile)
    {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (s : Schedule) (δ : ℝ) (decode : List (ℕ × Col) → List T)
    (ops : Stmt → List (Opening Bool)) (p : Params) (c : NetworkCheck.Period.Commitment) (x : Fin p.windows → Claims.Msg)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : (i : Fin p.windows) → Claims.Prover (WindowTape D x Ω i))
    (ev : PeriodCoins D x Ω → Evidence) : Prop :=
  Claims.Pr (fun ω : PeriodCoins D x Ω => PeriodAccepts (chk p c (ev ω)) ∧ ∃ i, ¬ IngressDeclared H D s ops δ (x i)) ≤
    periodErr H D δ decode ops x Ω P

/-! ## The guarantee -/

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The server's ingress over a period, at the check `chk`.** At every deployment `D` of the sampled verifier where
`Claims.Sampling.KS` holds, the trace's commitment binds and statements are their descriptors flattened, with a decoder
that lists every trace within Flock's radius, at `δ ≥ 0`, over a rows-form period `p` (schedule `s`) whose `W` windows'
statements `x i` are the scan's (`ScanScope`), with the openings `ops st` at each statement and the scan class `cls`:

* every period's provers (window `i`'s reading every other window's coins) whose answers are of record have
  `Pr[chk accepts ∧ some window's ingress isn't one the scan accepts] ≤ Σ_i (δ + ε_ks + ε_bind + Pr[B_tap])`;
* an outside sender passing secrets `Ms` into the server through the warden (the interaction `I` on the requests' link,
  `n = W · T` steps, under W2, W4 with allocation `r`, and J at `Dj`), whose every run keeps its statuses complete,
  whose record holds cells, whose windows' statements are `sw i` (partition `Pw i`) with the record's rows registered
  at the scan's gates, and whose secrets a decoder reads off the wire: its secrets whose every window's ingress the
  scan accepts number at most
  `∏_i Σ_(F faulty at δ) 2^(#ingress(F) · faultyIngressBits + windowBits) · (∏_ℓ (r ℓ + 1))^n · Dj^n`, where the
  scan counts requests: each faulty ingress unit costs its whole bucket's row and two requests a lane. -/
def PerServerIngressBy (chk : Params → NetworkCheck.Period.Commitment → Evidence → Core.Outcome Public Profile) :
    Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (hks : KS H D) (hb : D.commit.Binds H)
    (hflat : NetworkScan.StmtFlattens D) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (p : Params) (s : Schedule) (sd : ByteArray) (sc : Scan) (hform : p.form = .rows s sd sc) (cls : ScanClass)
    (ops : Stmt → List (Opening Bool)),
    (∀ (c : NetworkCheck.Period.Commitment) (x : Fin p.windows → Claims.Msg) (hx : ∀ i, ScanScope D s (x i))
        (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : (i : Fin p.windows) → Claims.Prover (WindowTape D x Ω i))
        (ev : PeriodCoins D x Ω → Evidence) (hrec : AnswersOfRecord H D p s sd c x P ev),
        PeriodSound chk H D s δ decode ops p c x Ω P ev) ∧
      ∀ {K : ℕ} {Z In Act Wire : Type} (I : Interaction 1 K s.allocation Z In Act Wire) (Dj : ℕ) (r : Fin 1 → ℕ)
        (view : ℕ → Obs 1 K s.allocation → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
        (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier Dj)
        (sw : Fin p.windows → Stmt) (Pw : Fin p.windows → Partition)
        (hcount : ∀ i, ScanCountsRequests s cls (sw i) (Pw i) δ)
        (Ms : Set Z) (xm : Z → Fin p.windows → Claims.Msg)
        (hsw : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st →
          st.prog = (sw i).prog ∧ st.desc = (sw i).desc ∧ st.design = (sw i).design ∧ D.partition st = some (Pw i))
        (hacc : ∀ m ∈ Ms, StatusComplete (p.windows * s.window) (I.obsRun m (p.windows * s.window)))
        (hcells : ∀ m ∈ Ms, RecordsCells s (I.obsRun m (p.windows * s.window)))
        (hpass : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → OpensPass H D.commit st (ops st))
        (hrows : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st →
          WardenRowsOpen s st (ops st) (windowIns s (I.obsRun m (p.windows * s.window)) i.val))
        (hok : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → OkOpens s st (ops st))
        (hchain : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → ChainOpens s st (ops st))
        (dec : List Wire → Z) (hdec : ∀ m ∈ Ms, dec (I.wireRun m (p.windows * s.window)) = m),
        {m | m ∈ Ms ∧ ∀ i, IngressDeclared H D s ops δ (xm m i)}.encard ≤
          (((∏ i, ∑ F ∈ faultySets (sw i).design (Pw i) δ,
              2 ^ (faultyIngress (Pw i) (layOf s (sw i)) s.window (checksOf s (sw i)).1 F * faultyIngressBits s cls +
                windowBits s cls)) *
            ((∏ ℓ, (r ℓ + 1)) ^ (p.windows * s.window) * Dj ^ (p.windows * s.window)) : ℕ) : ℕ∞)

/-- **The server's ingress over a rows-form period**: `PerServerIngressBy` at the check of record,
`NetworkCheck.Period.check`. -/
def PerServerIngress : Prop := PerServerIngressBy check

/-- **The count where the draw takes every ingress unit**: `PerServerIngress`' count where every window's draw takes
every ingress unit (`IngressDrawn`), so that no faulty set holds one: the secrets an outside sender passes into the
server through the warden, whose every window's ingress the scan accepts, number at most
`∏_i #faultySets · 2^windowBits · (∏_ℓ (r ℓ + 1))^n · Dj^n`, `log2 #faultySets + windowBits` bits a window. -/
def PerServerIngressDrawn : Prop :=
  ∀ {Coin Col T : Type} (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (δ : ℝ) (p : Params) (s : Schedule)
    (cls : ScanClass) (ops : Stmt → List (Opening Bool)) {K : ℕ} {Z In Act Wire : Type}
    (I : Interaction 1 K s.allocation Z In Act Wire) (Dj : ℕ) (r : Fin 1 → ℕ)
    (view : ℕ → Obs 1 K s.allocation → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier Dj)
    (sw : Fin p.windows → Stmt) (Pw : Fin p.windows → Partition)
    (hcount : ∀ i, ScanCountsRequests s cls (sw i) (Pw i) δ) (hdrawn : ∀ i, IngressDrawn s (sw i) (Pw i))
    (Ms : Set Z) (xm : Z → Fin p.windows → Claims.Msg)
    (hsw : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st →
      st.prog = (sw i).prog ∧ st.desc = (sw i).desc ∧ st.design = (sw i).design ∧ D.partition st = some (Pw i))
    (hacc : ∀ m ∈ Ms, StatusComplete (p.windows * s.window) (I.obsRun m (p.windows * s.window)))
    (hcells : ∀ m ∈ Ms, RecordsCells s (I.obsRun m (p.windows * s.window)))
    (hpass : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → OpensPass H D.commit st (ops st))
    (hrows : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st →
      WardenRowsOpen s st (ops st) (windowIns s (I.obsRun m (p.windows * s.window)) i.val))
    (hok : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → OkOpens s st (ops st))
    (hchain : ∀ m ∈ Ms, ∀ i st, D.parse (xm m i) = some st → ChainOpens s st (ops st))
    (dec : List Wire → Z) (hdec : ∀ m ∈ Ms, dec (I.wireRun m (p.windows * s.window)) = m),
    {m | m ∈ Ms ∧ ∀ i, IngressDeclared H D s ops δ (xm m i)}.encard ≤
      (((∏ i, (faultySets (sw i).design (Pw i) δ).card * 2 ^ windowBits s cls) *
        ((∏ ℓ, (r ℓ + 1)) ^ (p.windows * s.window) * Dj ^ (p.windows * s.window)) : ℕ) : ℕ∞)

/-! ## Its honest and nontrivial statements -/

/-- **Every window's and audit's answer is accepted, at one δ** (named assumption): the counts are the parameters',
every link-window's rows-form check accepts with its scan's facts at one `δ`, and every audit's check accepts. An honest
prover's period gives it where the service accepts every window's and audit's proof at one `δ`. -/
def HonestWindowsAccepted (p : Params) (s : Schedule) (sd : ByteArray) (c : NetworkCheck.Period.Commitment)
    (e : Evidence) : Prop :=
  c.windows.size = p.windows ∧ c.audits.size = p.audits.size ∧ e.windows.size = p.windows ∧
    e.audits.size = p.audits.size ∧
    (∃ δ0 : Core.Ratio, ∀ i < p.windows, ∃ reg x f, c.windows[i]? = some reg ∧ e.windows[i]? = some x ∧
      rowsWindowCheck p s sd (p.first + i) reg x = .ok f ∧ f.delta = δ0) ∧
    ∀ i < p.audits.size, ∃ a reg x f, p.audits[i]? = some a ∧ c.audits[i]? = some reg ∧ e.audits[i]? = some x ∧
      auditCheck i a reg x = .ok f

/-- **`PerServerIngress`' honest statement**: a rows-form period whose answers are accepted (`HonestWindowsAccepted`)
is accepted, every window answered. -/
def PerServerIngressHonest : Prop :=
  ∀ (p : Params) (s : Schedule) (sd : ByteArray) (sc : Scan) (c : NetworkCheck.Period.Commitment) (e : Evidence),
    p.form = .rows s sd sc → HonestWindowsAccepted p s sd c e →
      PeriodAccepts (check p c e) ∧ ∀ i < p.windows, WindowAnswered p s sd c e i

/-- `chk` accepts every period, whatever it publishes. -/
def PeriodAcceptsAll (chk : Params → NetworkCheck.Period.Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ p c e, PeriodAccepts (chk p c e)

/-- **`PerServerIngress`' nontrivial statement**: a check accepting every period is not sound for the period's ingress
at any period one of whose windows' registered ingress the scan doesn't accept, wherever the period's error is below
1. -/
def PerServerIngressNontrivial : Prop :=
  ∀ (chk : Params → NetworkCheck.Period.Commitment → Evidence → Core.Outcome Public Profile), PeriodAcceptsAll chk →
    ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      (H : Claims.Msg → Claims.Msg) (D : Deploy Coin Col T) (s : Schedule) (δ : ℝ) (decode : List (ℕ × Col) → List T)
      (ops : Stmt → List (Opening Bool)) (p : Params) (c : NetworkCheck.Period.Commitment)
      (x : Fin p.windows → Claims.Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
      (P : (i : Fin p.windows) → Claims.Prover (WindowTape D x Ω i)) (ev : PeriodCoins D x Ω → Evidence),
      (∃ i, ¬ IngressDeclared H D s ops δ (x i)) → periodErr H D δ decode ops x Ω P < 1 →
        ¬ PeriodSound chk H D s δ decode ops p c x Ω P ev

end Properties.NetworkCertifier
