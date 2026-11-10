module

public import Properties.Nci
public import Properties.Nci.Baseline
public import Flock.TemplateQuery
public import Spot.Wires
public import NetworkCheck.Period

/-!
# What the record form's acceptance must mean (LEAN-238): a draft, never elaborated

Half done, stopped at 08:19Z on 10 Nov: this file has not been through `lean_changed.py --try` once, it is in no root of
`Security/Properties/lakefile.toml`, and nothing here is proved. It records the statement as it stood, so the next worker
starts from it and not from the sources.

The form (training's `NciCheck/Record.lean`, TRN-76, not on `main`) is taken through `Form`, the functions it must
export. The record's circuit is the population Program's gates with every wired reader leaf replaced by the writer gates
whose bits it reads (`Lands`), θ the unwired leaves of the rows the form holds (`Form.heldBy`), and the inference blocks
the slots (`Form.slotOf`).

* `Complies`: acceptance gives a circuit the record lays out, and on every such circuit the rate rule
  (`Nci.CompliesRate`). No premise.
* `Provenance`: every row the form holds is a read or a root of a parameter the log's first record and the run's own
  register under one root, or an opened port of a reader whose closure digest is a pin. No premise.
* `DigestPins`: lemma (i). Two descriptors that give one Definition one closure digest decode it to the same gates, or
  two byte strings the two digests hashed collide under SHA-512. No assumption: the collision is in the conclusion.
* `PinnedOpens`: how (ii), PRF-458's P1, enters: as the premise `Opens P₀ k₀` at the reference Definition, proofs' to
  state and prove, carried to the record's reader by `DigestPins`.

Not settled, each the form's draft to answer: whether `j`'s `program` is the descriptor's text; how the trusted input
names the run (`TrustedRun` follows v0: by digest); how the run's record names the record's bytes (`RunNames` guesses
canonical JSON); who holds an opening's verdict to 1.
-/

@[expose] public section

namespace NciCheck.Properties.Record

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn Node)
open Flock.TemplateQuery (TRange)
open NciCheck.Baseline (getD strAt digest)
open NetworkCheck.Period (canonicalEntry definitionDigest slotEntryDigest)

/-! ## The interface -/

/-- What the form computes of a record, all `rate` reads: the greatest block's gates and fresh bits. -/
structure Summary where
  gates : ℕ
  fresh : ℕ

/-- A reader row: its template's descriptor id, the unit of that template (counted as the wires count), the
parameter, and the row of the parameter's cut. -/
structure Row where
  tid : String
  unit : ℕ
  port : String
  row : ℕ

/-- A pinned reader, the verifier's own and never the record's: its Definition's closure digest, the parameters an
opening registers (never the key), the parameters that take the table's root and domain id, and the return field that
is the verdict. -/
structure Pin where
  digest : String
  opened : List String
  root : String
  domain : String
  verdict : ℕ

/-- Why the form holds a row in θ: it reads the registered parameter `param`; it holds `param`'s table's root or domain
id; or it is an opened port of an opening of `param`'s table by a pinned reader. -/
inductive Held where
  | read (param : String)
  | root (param : String)
  | opened (param : String) (pin : Pin)

/-- The functions the form exports. `summarize pins j trusted` with `j = {program, record, table}` and
`trusted = {log, run}`; `rate s Γ num den` writes `complies`. -/
structure Form where
  wiringOf : Prog → Json → Except String Spot.Wires.Wiring
  slotOf : Json → String → ℕ → Option ℕ
  heldBy : List Pin → Json → Json → Row → Option Held
  summarize : List Pin → Json → Json → Except String Summary
  rate : Summary → ℕ → ℕ → ℕ → Json

/-! ## The record's circuit -/

/-- Unit `u` of template `tid` is the activation `sc`: member `m` of a root node of `tid`, after the units of `tid`'s
earlier root nodes. -/
def UnitAt (P : Prog) (rs : Array TRange) (tid : String) (u : ℕ) (sc : Scope) : Prop :=
  ∃ i m, i < rs.size ∧ rs[i]!.template = tid ∧ m < rs[i]!.count ∧
    u = ((rs.extract 0 i).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 + m ∧
    sc = P.child P.rootScope rs[i]!.node m

/-- The bit at which leaf `l` starts in a row whose leaves have the widths `ws`. -/
def bitAt (ws : Array ℕ) (l : ℕ) : ℕ := (ws.extract 0 l).foldl (· + ·) 0

/-- Row `row` of parameter `i` cut `k` ways: its first leaf and its leaves' widths. -/
def inRow (d : Defn) (i k row : ℕ) : ℕ × Array ℕ :=
  (row * (d.params[i]!.2.leaves / k),
    d.params[i]!.2.widths.extract (row * (d.params[i]!.2.leaves / k)) ((row + 1) * (d.params[i]!.2.leaves / k)))

/-- A return type's fields, as `interface.layout` takes them. -/
def fieldsOf : Ty → Array Ty
  | .tuple fs => fs
  | .record fs => fs.map (·.2)
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

/-- A wire covers Program gate `g`: it is a leaf of a reader row that has a writer. -/
def Wired (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (g : ℕ) : Prop :=
  ∃ tid u port row sc l lo hi, (S.writerOf tid u port row).isSome ∧ UnitAt P rs tid u sc ∧
    InLeaf P sc (W.cutsOf tid port) port row l g lo hi

/-- The wired gate `g` reads `g'`: `g'` is a leaf of the output row `g`'s row is wired to, and their bits meet. A wired
leaf past its writer's bits (the row's zero padding) reads nothing. -/
def Feeds (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (g' g : ℕ) : Prop :=
  ∃ tid u port row wt wu out sc sc' l l' lo hi lo' hi',
    S.writerOf tid u port row = some (wt, wu, out) ∧ 0 ≤ wu ∧ UnitAt P rs tid u sc ∧ UnitAt P rs wt wu.toNat sc' ∧
    InLeaf P sc (W.cutsOf tid port) port row l g lo hi ∧ OutLeaf P sc' out l' g' lo' hi' ∧ lo < hi' ∧ lo' < hi

/-- Where a read of Program gate `o` lands: `o` itself when no wire covers it, else through the wires, a writer's
pass-through of its own wired input followed in turn. -/
inductive Lands (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : ℕ → ℕ → Prop
  | here (o : ℕ) : ¬ Wired P rs W S o → Lands P rs W S o o
  | wire (o g a : ℕ) : Feeds P rs W S g o → Lands P rs W S g a → Lands P rs W S o a

/-- `D` is the wired circuit: its gates are the Program's gates no wire covers, gate `x` the Program's `at_ x`, with
that gate's width and kind, reading where its operands land. As `NciCheck.Properties.LaysOut`, operations and outputs
are free. -/
def LaysOut (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (D : Verity.Circuit)
    (at_ : Fin D.N → ℕ) : Prop :=
  Function.Injective at_ ∧ (∀ g, (∃ x, at_ x = g) ↔ g < gates P ∧ ¬ Wired P rs W S g) ∧
    ∀ x : Fin D.N, ∃ p ops sc, P.gate (at_ x) = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value (D.w x) ∧
      D.isInput x = Flock.Program.isInputId (P.defs[p]!).key ∧
      ∀ a : Fin D.N, a ∈ D.reads x ↔ ∃ o ∈ ops, Lands P rs W S o (at_ a)

/-- θ is the gates of the rows `held` holds. -/
def Names (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (held : Row → Bool) {D : Verity.Circuit}
    (at_ : Fin D.N → ℕ) (θ : Finset (Fin D.N)) : Prop :=
  ∀ x, x ∈ θ ↔ ∃ r sc l lo hi, held r = true ∧ UnitAt P rs r.tid r.unit sc ∧
    InLeaf P sc (W.cutsOf r.tid r.port) r.port r.row l (at_ x) lo hi

/-- The blocks are the slots: a computed gate lies in a unit, and its block is that unit's slot. -/
def Divides (P : Prog) (rs : Array TRange) (slot : String → ℕ → Option ℕ) {D : Verity.Circuit} (at_ : Fin D.N → ℕ)
    {n : ℕ} (U : Verity.Partition D n) : Prop :=
  ∀ x, D.isInput x = false → ∃ tid u sc s, UnitAt P rs tid u sc ∧ sc.off ≤ at_ x ∧
    at_ x < sc.off + (P.defs[sc.d]!).gates ∧ slot tid u = some s ∧ (U.unit x).map Fin.val = some s

/-- `d` is the record's circuit as the form reads it: its wiring passes `Spot.Wires.check`, its templates are the
Program's root nodes (`Flock.TemplateQuery.evaluate`), and `d` is the wired circuit with θ the held rows and `𝓘` the
slots. -/
def Served (F : Form) (pins : List Pin) (P : Prog) (record trusted : Json) (d : Verity.RegisteredCircuit) : Prop :=
  ∃ W S rs, ∃ at_ : Fin d.D.N → ℕ, F.wiringOf P record = .ok W ∧ Spot.Wires.check W = .ok S ∧
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs ∧
    LaysOut P rs W S d.D at_ ∧ Names P rs W (fun r => (F.heldBy pins record trusted r).isSome) at_ d.θ ∧
    Divides P rs (F.slotOf record) at_ d.𝓘

/-! ## The property -/

/-- The form accepts `j` and `trusted` at the verifier's own pins, `Γ` and `α = num / den`, on the Program `P` that
`j`'s `program` decodes to. -/
def Accepts (F : Form) (pins : List Pin) (Γ num den : ℕ) (P : Prog) (text : String) (j trusted : Json) : Prop :=
  strAt j "program" = some text ∧ Flock.Program.decode text.toUTF8 = .ok P ∧
    ∃ s, F.summarize pins j trusted = .ok s ∧ getD (F.rate s Γ num den) "complies" = .bool true

/-- **The form's acceptance gives the rate rule on the served record's circuit.** The record lays out a circuit, and
every circuit it lays out meets the policy block by block at `Γ` and `α`, so it complies. -/
def Complies (F : Form) : Prop :=
  ∀ (pins : List Pin) (Γ num den : ℕ) (P : Prog) (text : String) (j trusted : Json),
    Accepts F pins Γ num den P text j trusted →
    (∃ d, Served F pins P (getD j "record") trusted d) ∧
      ∀ d, Served F pins P (getD j "record") trusted d →
        (∀ u, Meets d d.𝓘 Γ ((num : ℝ) / den) u) ∧ Nci.CompliesRate Γ ((num : ℝ) / den) d

/-- `p` is θ: the log's first record and the run's own register it under one root (training's `isTheta`). -/
def IsTheta (log : Array Json) (run : Json) (p : String) : Prop :=
  0 < log.size ∧ ∃ R, strAt (getD (getD log[0]! "window") "registered_roots") p = some R ∧
    strAt (getD (getD run "window") "registered_roots") p = some R

/-- The trusted input holds the chained log `log` and names `run`, one of its records, by its digest. -/
def TrustedRun (tr : Json) (log : Array Json) (run : Json) : Prop :=
  getD tr "log" = .arr log ∧ Chained log ∧ ∃ n < log.size, log[n]! = run ∧ ∃ r, digest run = .ok r ∧ strAt tr "run" = some r

/-- The run's record names the Program and the record by their SHA-512. -/
def RunNames (run : Json) (P : Prog) (record : Json) : Prop :=
  strAt run "program" = some (Flock.hexOf P.sha512) ∧ ∃ s, Flock.canon record = .ok s ∧
    strAt (getD run "window") "circuit" = some (Flock.hexOf (Flock.Sha512.hash s.toUTF8))

/-- The reason the form gives for holding row `r` is good against the log, the run and the pins. An opened port is one
the pin lists, of a reader whose closure digest in the Program's descriptor `pj` is the pin's, with its root and domain
ports holding the table's. -/
def Good (F : Form) (pins : List Pin) (pj record trusted : Json) (log : Array Json) (run : Json) (r : Row) :
    Held → Prop
  | .read p => IsTheta log run p
  | .root p => IsTheta log run p
  | .opened p pin => IsTheta log run p ∧ pin ∈ pins ∧ slotEntryDigest pj r.tid = .ok pin.digest ∧
      r.port ∈ pin.opened ∧ F.heldBy pins record trusted ⟨r.tid, r.unit, pin.root, 0⟩ = some (.root p) ∧
      F.heldBy pins record trusted ⟨r.tid, r.unit, pin.domain, 0⟩ = some (.root p)

/-- **Every row the form holds is registered.** On acceptance the trusted input's run names this Program and record,
and each held row's reason is `Good`. -/
def Provenance (F : Form) : Prop :=
  ∀ (pins : List Pin) (Γ num den : ℕ) (P : Prog) (text : String) (pj j trusted : Json),
    Accepts F pins Γ num den P text j trusted → Flock.JsonParse.parse text = .ok pj →
    ∃ log run, TrustedRun trusted log run ∧ RunNames run P (getD j "record") ∧
      ∀ r h, F.heldBy pins (getD j "record") trusted r = some h →
        Good F pins pj (getD j "record") trusted log run r h

/-! ## Lemma (i): the closure digest pins the decoded Definition

Against today's names in `NetworkCheck.Period` (`flatParts`, `written`, `canonicalEntry`, `definitionDigest`,
`slotEntryDigest`); NET-301 renames them into `NciCheck.DefinitionDigest`, `slotEntryDigest` to `closureDigest`. -/

/-- `j` is the descriptor the bytes `b` hold, as `Flock.Program.decode` parses them. -/
def Descriptor (b : ByteArray) (j : Json) : Prop :=
  ∃ text, String.fromUTF8? b = some text ∧ Flock.JsonParse.parse text = .ok j

/-- A descriptor's interning table `k`, as `slotEntryDigest` reads it. -/
def tab (j : Json) (k : String) : Array Json := match j.getObjValD k with
  | .arr xs => xs
  | _ => #[]

/-- A descriptor entry's nodes, as `slotEntryDigest` reads them. -/
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
maps of those Definitions' ids to their digests. Finitely many, so a collision among them is not for free. -/
def Hashed (j : Json) (key : String) (x : ByteArray) : Prop :=
  (∃ k e s, Reaches j key k ∧ canonicalEntry (tab j "callees") (tab j "types") (entry j k) = .ok e ∧
      Flock.canon e = .ok s ∧ x = s.toUTF8) ∨
    ∃ (seen : List (String × Json)) (s : String),
      (∀ p ∈ seen, Reaches j key p.1 ∧
        ∃ d, definitionDigest (tab j "callees") (tab j "types") (entry j p.1) = .ok d ∧ p.2 = .str d) ∧
      Flock.canon (Json.mkObj seen) = .ok s ∧ x = s.toUTF8

/-- Definition `k` of `P` and `k'` of `P'` are one Definition: one descriptor id, kind, signature and count, and, node
by node, one body whose callees are one Definition in turn. Only a callee's place in each decoder's order may differ. -/
inductive Same (P P' : Prog) : ℕ → ℕ → Prop
  | intro (k k' : ℕ) (d d' : Defn) : P.defs[k]? = some d → P'.defs[k']? = some d' →
      d.key = d'.key → d.id = d'.id → d.prim = d'.prim → d.params = d'.params → d.ret = d'.ret →
      d.binding = d'.binding → d.gates = d'.gates → d.paramLeaves = d'.paramLeaves → d.outLeaves = d'.outLeaves →
      d.body.params = d'.body.params → d.body.retType = d'.body.retType → d.body.ret = d'.body.ret →
      d.body.offsets = d'.body.offsets → d.body.nodes.size = d'.body.nodes.size →
      (∀ n < d.body.nodes.size, ({ d.body.nodes[n]! with fn := 0 } : Node) = { d'.body.nodes[n]! with fn := 0 } ∧
        Same P P' d.body.nodes[n]!.fn d'.body.nodes[n]!.fn) →
      Same P P' k k'

/-- **Lemma (i).** Two Program files that give `key` one closure digest decode it to the same Definition, callees and
all, or two byte strings the two digests hashed are different and have one SHA-512. -/
def DigestPins : Prop :=
  ∀ (b b' : ByteArray) (j j' : Json) (P P' : Prog) (key h : String) (k k' : ℕ),
    Flock.Program.decode b = .ok P → Flock.Program.decode b' = .ok P' → Descriptor b j → Descriptor b' j' →
    slotEntryDigest j key = .ok h → slotEntryDigest j' key = .ok h → P.index[key]? = some k → P'.index[key]? = some k' →
    Same P P' k k' ∨
      ∃ x x', Hashed j key x ∧ Hashed j' key x' ∧ x ≠ x' ∧ Flock.Sha512.hash x = Flock.Sha512.hash x'

/-! ## How (ii) enters -/

/-- `Opens` reads only a Definition's gates: it holds of one Definition in any Program. -/
def OfGates (Opens : Prog → ℕ → Prop) : Prop :=
  ∀ P P' k k', Same P P' k k' → Opens P k → Opens P' k'

/-- **A held opening's reader is the reference Definition.** `Opens` is proofs' statement that a reader's verdict is 1
only when its ports open (PRF-458's P1), an open premise held at the reference Definition `key` of `P₀`, whose closure
digest is the pin's. Then the reader of every opened port the form holds under that pin satisfies it in the record's
own Program, or two byte strings the two digests hashed collide. -/
def PinnedOpens (F : Form) : Prop :=
  ∀ (Opens : Prog → ℕ → Prop) (b₀ : ByteArray) (j₀ : Json) (P₀ : Prog) (key : String) (k₀ : ℕ) (pin : Pin),
    OfGates Opens → Flock.Program.decode b₀ = .ok P₀ → Descriptor b₀ j₀ → slotEntryDigest j₀ key = .ok pin.digest →
    P₀.index[key]? = some k₀ → Opens P₀ k₀ →
    ∀ (pins : List Pin) (Γ num den : ℕ) (P : Prog) (text : String) (pj j trusted : Json) (r : Row) (p : String),
      Accepts F pins Γ num den P text j trusted → Flock.JsonParse.parse text = .ok pj → r.tid = key →
      F.heldBy pins (getD j "record") trusted r = some (.opened p pin) →
      (∃ k, P.index[key]? = some k ∧ Opens P k) ∨
        ∃ x x', Hashed j₀ key x ∧ Hashed pj key x' ∧ x ≠ x' ∧ Flock.Sha512.hash x = Flock.Sha512.hash x'

end NciCheck.Properties.Record
