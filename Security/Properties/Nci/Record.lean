module

public import Properties.Nci
public import Properties.Nci.Baseline
public import NciCheck.DefinitionDigest
public import Flock.TemplateQuery
public import Spot.Wires

/-!
# What accepting an execution phase's circuit record means

An execution phase's Program is a population of template instances: its root's nodes are calls and batches of
templates, and every unit's parameters are input gates of its own. Which unit feeds which, and which rows hold
registered data, are declarations of the phase's circuit record. A check of such a record against the inference-only
policy is taken here through `Form`, the functions it exports, so the statements hold a check to them before it is
written and read the same on any check that exports them.

**The record's circuit** (`Served`). Its gates are the Program's gates that no wire covers; a gate that read a wired
input leaf reads instead the writer's gates whose bits that leaf holds, a writer's pass-through of its own wired input
followed in turn (`Lands`). Its registered inputs θ are the leaves of the rows the check holds (`Form.held`), and its
inference blocks are the slots (`Form.slot`): a block is a set of units, and a root node that batches the units of
several slots lies in several blocks. A record lays out a circuit only where every unit's rows are its own input
gates (`OwnRows`) and every wire lands whole (`Aligned`: a writer's leaf lies inside one leaf of the row that reads it).
`Spot.Wires.check` compares a wire's two rows in words alone, so a check that stops there leaves a 16-bit reader leaf
free to read sixteen gates of 16 bits through a writer that passes one-bit leaves on.

**The rate rule** (`Complies`): where the check accepts, the instance's fields are the ones it judged by, the record
lays out a circuit, and every circuit it lays out meets the policy block by block at the instance's `Γ` and `α`, so
it complies (`Nci.CompliesRate`). It follows from four statements about the check, `Owns`, `LaidOut`, `Counts` and
`Rates` (`Proofs.Properties.Nci.Record`).

**Provenance** (`Provenance`): the run's own record names this Program and this record by their SHA-512s, and every
row the check holds is a registered read or a table read of a parameter that the log's first record registers under
the root the run commits it under, or a port of an opening whose reader is pinned (`Good`). A pin is the verifier's
own, never the record's: its Definition's closure digest (`NciCheck.DefinitionDigest.closureDigest`) and its ports by
role, so that an opening's key is never a row it registers.

**A pin fixes the Definition** (`DigestPins`): two Program texts that give a Definition one closure digest decode it
to the same Definition, callees and all (`Same`), or two of the byte strings the two digests hashed (`Hashed`) differ
and have one SHA-512. The collision is among those strings and no others: over all byte strings one exists by
counting, and a statement that offered it would say nothing.

**What a pinned reader opens** (`PinnedOpens`): a statement about a pinned Definition that reads only the Definition
(`OfDefinition`), held of a reference Program's copy, holds of the reader of every opened row the check holds under
that pin, up to such a collision. It follows from `DigestPins` and `Provenance`.

What these do not say: that a unit's verdict output is 1 (a statement about the committed values, the proof block's);
that equal roots hold equal rows, or that the data the log's first record registers was public
(`Nci.Assumptions.ThetaRegistered`).
-/

@[expose] public section

namespace NciCheck.Properties.Record

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn Node)
open Flock.TemplateQuery (TRange)
open NciCheck.Baseline (getD strAt digest lastWith)
open NciCheck.DefinitionDigest (canonicalEntry definitionDigest closureDigest)

/-! ## The interface -/

/-- A reader row: its template's descriptor id, the unit of that template (counted as the wires count units), the
parameter, and the row of the parameter's cut. -/
structure Row where
  tid : String
  unit : ℕ
  port : String
  row : ℕ

/-- A pinned reader of a table opening, the verifier's own: its Definition's closure digest, and its ports by role.
`key` is the parameter that says which row is opened, `opened` the parameters an opening registers (the row, its salt
and its path, never the key), `root` and `domain` the parameters that take the table's root and domain id, and
`verdict` the output row that holds the verdict. -/
structure Pin where
  digest : String
  key : String
  opened : List String
  root : String
  domain : String
  verdict : String

/-- Why the check holds a row in θ: it is a registered read of the parameter `param`; a table read of the root
(`root`), else the domain id, of the table `name` over `param`; or a port of an opening of that table by a reader
pinned by `pin`. -/
inductive Held where
  | read (param : String)
  | table (name param : String) (root : Bool)
  | opened (name param : String) (pin : Pin)

/-- An instance's fields: the Program's text, the record's text, `Γ`, `α = num / den`, and the pins. -/
structure Instance where
  program : String
  record : String
  gamma : ℕ
  num : ℕ
  den : ℕ
  pins : List Pin

/-- What the rate rule reads of a record: a block's gates, and the most fresh bits a block reads. -/
structure Summary where
  gates : ℕ
  fresh : ℕ

/-- The functions a check of a circuit record exports. `read j tr` reads the instance `j` and the trusted input `tr`
(the registration log and the digest of the run's own record), or refuses. Of a record it has read: `inst` is the
instance's fields, `wiring` the record's wires as `Spot.Wires.check` takes them, `slot tid u` the block of unit `u` of
template `tid`, `held r` whether row `r` counts as θ and why, and `summarize` the summary. `rate Γ num den s` is the
rate rule on a summary, `none` where it complies. -/
structure Form where
  Case : Type
  read : Json → Json → Except String Case
  inst : Case → Instance
  wiring : Case → Spot.Wires.Wiring
  slot : Case → String → ℕ → Option ℕ
  held : Case → Row → Option Held
  summarize : Case → Summary
  rate : ℕ → ℕ → ℕ → Summary → Option String

/-! ## The record's circuit -/

/-- Unit `u` of template `tid` is the activation `sc`: member `m` of a root node of `tid`, after the units of `tid`'s
earlier root nodes. -/
def UnitAt (P : Prog) (rs : Array TRange) (tid : String) (u : ℕ) (sc : Scope) : Prop :=
  ∃ i m, i < rs.size ∧ rs[i]!.template = tid ∧ m < rs[i]!.count ∧
    u = ((rs.extract 0 i).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 + m ∧
    sc = P.child P.rootScope rs[i]!.node m

/-- The bit at which leaf `l` starts in a row whose leaves have the widths `ws`: a row is its leaves' bits, in leaf
order. -/
def bitAt (ws : Array ℕ) (l : ℕ) : ℕ := (ws.extract 0 l).foldl (· + ·) 0

/-- Row `row` of parameter `i` cut `k` ways: its first leaf and its leaves' widths. -/
def inRow (d : Defn) (i k row : ℕ) : ℕ × Array ℕ :=
  (row * (d.params[i]!.2.leaves / k),
    d.params[i]!.2.widths.extract (row * (d.params[i]!.2.leaves / k)) ((row + 1) * (d.params[i]!.2.leaves / k)))

/-- A return type's fields, each one output row: a tuple's or a record's fields, else the type itself. -/
def fieldsOf : Ty → Array Ty
  | .tuple fs => if fs.isEmpty then #[.tuple fs] else fs
  | .record fs => if fs.isEmpty then #[.record fs] else fs.map (·.2)
  | t => #[t]

/-- Output row `o<f>`: its first leaf and its leaves' widths. -/
def outRow (d : Defn) (f : ℕ) : ℕ × Array ℕ :=
  (((fieldsOf d.ret).extract 0 f).foldl (fun a t => a + t.leaves) 0, (fieldsOf d.ret)[f]!.widths)

/-- Leaf `l` of reader row `(port, row)` of the activation `sc`, the port cut `k` ways, is Program gate `g`, at bits
`[lo, hi)` of the row. -/
def InLeaf (P : Prog) (sc : Scope) (k : ℕ) (port : String) (row l g lo hi : ℕ) : Prop :=
  ∃ i, i < (P.defs[sc.d]!).params.size ∧ (P.defs[sc.d]!).params[i]!.1 = port ∧ row < k ∧
    l < (inRow (P.defs[sc.d]!) i k row).2.size ∧
    Flock.Program.resolve P sc (⟨true, i, (inRow (P.defs[sc.d]!) i k row).1 + l⟩ : Ref) = .ok g ∧
    lo = bitAt (inRow (P.defs[sc.d]!) i k row).2 l ∧ hi = lo + (inRow (P.defs[sc.d]!) i k row).2[l]!

/-- Leaf `l` of output row `out` of the activation `sc` is Program gate `g`, at bits `[lo, hi)` of the row. -/
def OutLeaf (P : Prog) (sc : Scope) (out : String) (l g lo hi : ℕ) : Prop :=
  ∃ f, out = "o" ++ toString f ∧ f < (fieldsOf (P.defs[sc.d]!).ret).size ∧
    l < (outRow (P.defs[sc.d]!) f).2.size ∧
    Flock.Program.resolve P sc ((P.defs[sc.d]!).body.ret.get ((outRow (P.defs[sc.d]!) f).1 + l)) = .ok g ∧
    lo = bitAt (outRow (P.defs[sc.d]!) f).2 l ∧ hi = lo + (outRow (P.defs[sc.d]!) f).2[l]!

/-- Leaf `l` of the reader row `r` is Program gate `g`, at bits `[lo, hi)` of the row. -/
def RowLeaf (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (r : Row) (l g lo hi : ℕ) : Prop :=
  ∃ sc, UnitAt P rs r.tid r.unit sc ∧ InLeaf P sc (W.cutsOf r.tid r.port) r.port r.row l g lo hi

/-- **Every unit's rows are its own input gates**: a leaf of a reader row is an input gate, and no gate is a leaf of
two rows, or two leaves of one. -/
def OwnRows (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi →
      ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ Flock.Program.isInputId (P.defs[p]!).key = true) ∧
    ∀ r l lo hi r' l' lo' hi' g, RowLeaf P rs W r l g lo hi → RowLeaf P rs W r' l' g lo' hi' → r = r' ∧ l = l'

/-- A wire covers Program gate `g`: it is a leaf of a reader row that has a writer. -/
def Wired (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (g : ℕ) : Prop :=
  ∃ r l lo hi, (S.writerOf r.tid r.unit r.port r.row).isSome ∧ RowLeaf P rs W r l g lo hi

/-- Leaf `l'` of the output row that the reader row `r` is wired to is Program gate `g'`, at bits `[lo', hi')`. -/
def WriterLeaf (P : Prog) (rs : Array TRange) (S : Spot.Wires.Summary) (r : Row) (l' g' lo' hi' : ℕ) : Prop :=
  ∃ wt wu out sc', S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) ∧ 0 ≤ wu ∧
    UnitAt P rs wt wu.toNat sc' ∧ OutLeaf P sc' out l' g' lo' hi'

/-- **Every wire lands whole**: each leaf of a writer's output row lies inside one leaf of the row that reads it. -/
def Aligned (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  ∀ r l' g' lo' hi', WriterLeaf P rs S r l' g' lo' hi' → ∃ l g lo hi, RowLeaf P rs W r l g lo hi ∧ lo ≤ lo' ∧ hi' ≤ hi

/-- The wired gate `g` reads `g'`: `g'` is a leaf of the output row `g`'s row is wired to, and their bits meet. A wired
leaf past its writer's bits, the row's zero padding, reads nothing. -/
def Feeds (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (g' g : ℕ) : Prop :=
  ∃ r l l' lo hi lo' hi', RowLeaf P rs W r l g lo hi ∧ WriterLeaf P rs S r l' g' lo' hi' ∧ lo < hi' ∧ lo' < hi

/-- Where a read of Program gate `o` lands: `o` itself when no wire covers it, else through the wires, a writer's
pass-through of its own wired input followed in turn. -/
inductive Lands (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : ℕ → ℕ → Prop
  | here (o : ℕ) : ¬ Wired P rs W S o → Lands P rs W S o o
  | wire (o g a : ℕ) : Feeds P rs W S g o → Lands P rs W S g a → Lands P rs W S o a

/-- `D` is the wired circuit: its gates are the Program's gates no wire covers, gate `x` the Program's `at_ x`, with
that gate's width and kind, reading where its operands land. As in `NciCheck.Properties.LaysOut`, operations and
outputs are free. -/
def LaysOut (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (D : Verity.Circuit)
    (at_ : Fin D.N → ℕ) : Prop :=
  Function.Injective at_ ∧ (∀ g, (∃ x, at_ x = g) ↔ g < gates P ∧ ¬ Wired P rs W S g) ∧
    ∀ x : Fin D.N, ∃ p ops sc, P.gate (at_ x) = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value (D.w x) ∧
      D.isInput x = Flock.Program.isInputId (P.defs[p]!).key ∧
      ∀ a : Fin D.N, a ∈ D.reads x ↔ ∃ o ∈ ops, Lands P rs W S o (at_ a)

/-- θ is the gates of the rows `held` holds. -/
def Names (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (held : Row → Bool) {D : Verity.Circuit}
    (at_ : Fin D.N → ℕ) (θ : Finset (Fin D.N)) : Prop :=
  ∀ x, x ∈ θ ↔ ∃ r l lo hi, held r = true ∧ RowLeaf P rs W r l (at_ x) lo hi

/-- The blocks are the slots: a computed gate lies in a unit, and its block is that unit's slot. -/
def Divides (P : Prog) (rs : Array TRange) (slot : String → ℕ → Option ℕ) {D : Verity.Circuit} (at_ : Fin D.N → ℕ)
    {n : ℕ} (U : Verity.Partition D n) : Prop :=
  ∀ x, D.isInput x = false → ∃ tid u sc s, UnitAt P rs tid u sc ∧ sc.off ≤ at_ x ∧
    at_ x < sc.off + (P.defs[sc.d]!).gates ∧ slot tid u = some s ∧ (U.unit x).map Fin.val = some s

/-- **`d` is the circuit the record `c` lays out.** The instance's Program decodes; the record's wiring passes
`Spot.Wires.check` and its templates are the Program's root nodes (`Flock.TemplateQuery.evaluate`); every unit's rows
are its own input gates and every wire lands whole; and `d` is the wired circuit, θ the rows the check holds and `𝓘`
the slots. -/
def Served (F : Form) (c : F.Case) (d : Verity.RegisteredCircuit) : Prop :=
  ∃ (P : Prog) (S : Spot.Wires.Summary) (rs : Array TRange) (at_ : Fin d.D.N → ℕ),
    Flock.Program.decode (F.inst c).program.toUTF8 = .ok P ∧ Spot.Wires.check (F.wiring c) = .ok S ∧
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P)
      ((F.wiring c).templates.map (·.1)).toArray = .ok rs ∧
    OwnRows P rs (F.wiring c) ∧ Aligned P rs (F.wiring c) S ∧ LaysOut P rs (F.wiring c) S d.D at_ ∧
    Names P rs (F.wiring c) (fun r => (F.held c r).isSome) at_ d.θ ∧ Divides P rs (F.slot c) at_ d.𝓘

/-! ## The rate rule -/

/-- The record `c` carries the instance `j`'s own fields: the Program's and the record's texts, `gamma` a natural, and
`alpha` a pair of naturals `[num, den]` with `den > 0`. -/
def Carries (F : Form) (j : Json) (c : F.Case) : Prop :=
  j.getObjVal? "program" = .ok (.str (F.inst c).program) ∧ j.getObjVal? "record" = .ok (.str (F.inst c).record) ∧
    (∃ v, j.getObjVal? "gamma" = .ok v ∧ natOf? v = some (F.inst c).gamma) ∧
    (∃ a b, j.getObjVal? "alpha" = .ok (.arr #[a, b]) ∧ natOf? a = some (F.inst c).num ∧
      natOf? b = some (F.inst c).den) ∧ 0 < (F.inst c).den

/-- The check accepts the instance `j` and the trusted input `tr`, reading the record `c`: it reads them, and the
rate rule complies on the record's summary at the instance's `Γ` and `α`. -/
def Accepts (F : Form) (j tr : Json) (c : F.Case) : Prop :=
  F.read j tr = .ok c ∧ F.rate (F.inst c).gamma (F.inst c).num (F.inst c).den (F.summarize c) = none

/-- **A record the check reads carries its instance's fields.** -/
def Owns (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), F.read j tr = .ok c → Carries F j c

/-- **A record the check reads lays out a circuit.** -/
def LaidOut (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), F.read j tr = .ok c → ∃ d, Served F c d

/-- **The summary counts every block.** In every circuit a record the check reads lays out, each block reads at most
the summary's fresh bits, and has the summary's gates or none. -/
def Counts (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), F.read j tr = .ok c → ∀ d, Served F c d → ∀ u,
    Nci.novelReads d (d.𝓘.gates u) ≤ (F.summarize c).fresh ∧
      (d.D.size (d.𝓘.gates u) = (F.summarize c).gates ∨ d.𝓘.gates u = ∅)

/-- **The rate rule on a summary**: where it complies, a block's gates are at most `Γ` and its fresh bits at most
`num / den` times its gates. -/
def Rates (F : Form) : Prop :=
  ∀ (Γ num den : ℕ) (s : Summary), F.rate Γ num den s = none → s.gates ≤ Γ ∧ s.fresh * den ≤ num * s.gates

/-- **Accepting a record gives the rate rule on its circuit.** Where the check accepts, the record carries the
instance's fields and lays out a circuit, and every circuit it lays out meets the policy block by block at the
instance's `Γ` and `α = num / den`, so it complies. -/
def Complies (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), Accepts F j tr c →
    Carries F j c ∧ (∃ d, Served F c d) ∧
      ∀ d, Served F c d →
        (∀ u, Meets d d.𝓘 (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) u) ∧
          Nci.CompliesRate (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) d

/-! ## Provenance -/

/-- The registered parameter `p` is θ: the log's first record registers a value of its name (its last entry of that
name) under the root the run's own record commits `p` under. -/
def IsTheta (log : Array Json) (run : Json) (p : String) : Prop :=
  0 < log.size ∧ ∃ e R, lastWith log[0]! "registered" "value" p = some e ∧ strAt e "root" = some R ∧
    strAt (getD (getD run "window") "registered_roots") p = some R

/-- The trusted input `tr` holds the registration log `log`, in R6's chain (`Chained`), and names `run`, one of its
records, by its digest. -/
def TrustedRun (tr : Json) (log : Array Json) (run : Json) : Prop :=
  getD tr "log" = .arr log ∧ Chained log ∧
    ∃ n < log.size, log[n]! = run ∧ ∃ r, digest run = .ok r ∧ strAt tr "run" = some r

/-- The run's record names the Program `P` and the record's text by their SHA-512s. -/
def RunNames (run : Json) (P : Prog) (record : String) : Prop :=
  strAt run "program" = some (Flock.hexOf P.sha512) ∧
    strAt (getD run "window") "circuit" = some (Flock.hexOf (Flock.Sha512.hash record.toUTF8))

/-- The reason the check gives for holding row `r` of the record `c` is good, against the log, the run and the
Program's descriptor `pj`. A read's parameter is θ. An opened port is one its pin lists, the pin is the instance's,
the reader's closure digest in `pj` is the pin's, and the same unit's root and domain ports, by the pin's roles, are
table reads of the same table's root and domain id. -/
def Good (F : Form) (c : F.Case) (pj : Json) (log : Array Json) (run : Json) (r : Row) : Held → Prop
  | .read p => IsTheta log run p
  | .table _ p _ => IsTheta log run p
  | .opened name p pin => IsTheta log run p ∧ pin ∈ (F.inst c).pins ∧ closureDigest pj r.tid = .ok pin.digest ∧
      r.port ∈ pin.opened ∧
      (∀ row < (F.wiring c).cutsOf r.tid pin.root,
        F.held c ⟨r.tid, r.unit, pin.root, row⟩ = some (.table name p true)) ∧
      ∀ row < (F.wiring c).cutsOf r.tid pin.domain,
        F.held c ⟨r.tid, r.unit, pin.domain, row⟩ = some (.table name p false)

/-- **Every row the check holds is registered.** For a record the check reads, the trusted input's run names its
Program and its text, and each held row's reason is `Good`. -/
def Provenance (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case) (P : Prog) (pj : Json), F.read j tr = .ok c →
    Flock.Program.decode (F.inst c).program.toUTF8 = .ok P → Flock.JsonParse.parse (F.inst c).program = .ok pj →
    ∃ log run, TrustedRun tr log run ∧ RunNames run P (F.inst c).record ∧
      ∀ r h, F.held c r = some h → Good F c pj log run r h

/-! ## A pin fixes the Definition -/

/-- A descriptor's interning table `k`, as `closureDigest` reads it. -/
def tab (j : Json) (k : String) : Array Json := match j.getObjValD k with
  | .arr xs => xs
  | _ => #[]

/-- A descriptor entry's nodes, as `closureDigest` reads them. -/
def nodesOf (e : Json) : Array Json := match (e.getObjValD "body").getObjValD "nodes" with
  | .arr xs => xs
  | _ => #[]

/-- Definition `k`'s entry in the descriptor `j`. -/
def entry (j : Json) (k : String) : Json := (j.getObjValD "definitions").getObjValD k

/-- `key` reaches `k` in the descriptor `j`: itself, or a node's callee in a composite Definition it reaches. -/
inductive Reaches (j : Json) (key : String) : String → Prop
  | self : Reaches j key key
  | callee (k c : String) (n : Json) : Reaches j key k → (entry j k).getObjValD "form" = .str "composite" →
      n ∈ nodesOf (entry j k) →
      ((n.getObjValD "fn").getNat?.toOption.bind fun i => (tab j "callees")[i]?) = some (.str c) → Reaches j key c

/-- The byte strings `key`'s closure digest in `j` hashes: the canonical entry of each Definition `key` reaches, and
the object that maps exactly those Definitions' ids to their digests. -/
def Hashed (j : Json) (key : String) (x : ByteArray) : Prop :=
  (∃ k e s, Reaches j key k ∧ canonicalEntry (tab j "callees") (tab j "types") (entry j k) = .ok e ∧
      Flock.canon e = .ok s ∧ x = s.toUTF8) ∨
    ∃ (seen : List (String × Json)) (s : String),
      (∀ k, Reaches j key k ↔ ∃ p ∈ seen, p.1 = k) ∧
      (∀ p ∈ seen, ∃ d, definitionDigest (tab j "callees") (tab j "types") (entry j p.1) = .ok d ∧ p.2 = .str d) ∧
      Flock.canon (Json.mkObj seen) = .ok s ∧ x = s.toUTF8

/-- Two of the byte strings that `key`'s closure digests in `j` and in `j'` hash differ and have one SHA-512. -/
def Collides (j j' : Json) (key : String) : Prop :=
  ∃ x x', Hashed j key x ∧ Hashed j' key x' ∧ x ≠ x' ∧ Flock.Sha512.hash x = Flock.Sha512.hash x'

/-- Definition `k` of `P` and `k'` of `P'` are one Definition: one descriptor id, kind, signature and count, and, node
by node, one body whose callees are one Definition in turn. Only a callee's place in each decoder's order may differ. -/
inductive Same (P P' : Prog) : ℕ → ℕ → Prop
  | intro (k k' : ℕ) (d d' : Defn) : P.defs[k]? = some d → P'.defs[k']? = some d' →
      d.key = d'.key → d.id = d'.id → d.prim = d'.prim → d.params = d'.params → d.ret = d'.ret →
      d.binding = d'.binding → d.gates = d'.gates → d.paramLeaves = d'.paramLeaves → d.outLeaves = d'.outLeaves →
      d.body.params = d'.body.params → d.body.retType = d'.body.retType → d.body.ret = d'.body.ret →
      d.body.offsets = d'.body.offsets → d.body.nodes.size = d'.body.nodes.size →
      (∀ n < d.body.nodes.size, ({ d.body.nodes[n]! with fn := 0 } : Node) = { d'.body.nodes[n]! with fn := 0 }) →
      (∀ n < d.body.nodes.size, Same P P' d.body.nodes[n]!.fn d'.body.nodes[n]!.fn) →
      Same P P' k k'

/-- **The closure digest pins the decoded Definition.** Two Program texts that give `key` one closure digest decode it
to the same Definition, callees and all, or two byte strings the two digests hashed differ and have one SHA-512. -/
def DigestPins : Prop :=
  ∀ (text text' : String) (j j' : Json) (P P' : Prog) (key h : String) (k k' : ℕ),
    Flock.Program.decode text.toUTF8 = .ok P → Flock.Program.decode text'.toUTF8 = .ok P' →
    Flock.JsonParse.parse text = .ok j → Flock.JsonParse.parse text' = .ok j' →
    closureDigest j key = .ok h → closureDigest j' key = .ok h →
    P.index[key]? = some k → P'.index[key]? = some k' →
    Same P P' k k' ∨ Collides j j' key

/-! ## What a pinned reader opens -/

/-- `Opens` reads only the Definition: it holds of one Definition in any Program. -/
def OfDefinition (Opens : Pin → Prog → ℕ → Prop) : Prop :=
  ∀ pin P P' k k', Same P P' k k' → Opens pin P k → Opens pin P' k'

/-- **The reader of a held opening is the pinned Definition.** `Opens pin P k` is a statement about Definition `k` of
`P` with the pin's roles: that its verdict is 1 only when its opened ports open to the root on its root port, at its
key. Let it read only the Definition, and hold of `key` in a reference Program whose closure digest of `key` is the
pin's. Then it holds of the reader `key` of every opened row the check holds under that pin, in the instance's own
Program, or two byte strings the two closure digests hashed differ and have one SHA-512. -/
def PinnedOpens (F : Form) : Prop :=
  ∀ (Opens : Pin → Prog → ℕ → Prop) (text₀ : String) (j₀ : Json) (P₀ : Prog) (k₀ : ℕ) (key : String) (pin : Pin),
    OfDefinition Opens → Flock.Program.decode text₀.toUTF8 = .ok P₀ → Flock.JsonParse.parse text₀ = .ok j₀ →
    closureDigest j₀ key = .ok pin.digest → P₀.index[key]? = some k₀ → Opens pin P₀ k₀ →
    ∀ (j tr : Json) (c : F.Case) (P : Prog) (pj : Json) (r : Row) (name p : String), F.read j tr = .ok c →
      Flock.Program.decode (F.inst c).program.toUTF8 = .ok P → Flock.JsonParse.parse (F.inst c).program = .ok pj →
      r.tid = key → F.held c r = some (.opened name p pin) →
      (∀ k, P.index[key]? = some k → Opens pin P k) ∨ Collides j₀ pj key

end NciCheck.Properties.Record
