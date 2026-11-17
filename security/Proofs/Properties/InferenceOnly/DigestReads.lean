module

public import Proofs.Properties.InferenceOnly.DigestPins
import Batteries.Tactic.OpenPrivate
import Circuits.Program

/-! `EntriesPin` from the decoder's invariant `DecodeReads` and the per-Definition step `ReadsPin`, by strong induction
on the index; both are stated and not proved. -/

public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program
open Claims.Json (Wf)
open FlockDeploy (JsonEquiv)
open InferenceOnlyCheck.DefinitionDigest (canonicalEntry)
open private natJ? intJ? paramsOf at_ buildNode from Circuits.Program

def calleeAt (cs : Array Json) (nd : Json) : Option Json :=
  (nd.getObjValD "fn").getNat?.toOption.bind fun x => cs[x]?

def formOf : String → Option Flock.Program.Form
  | "primitive" => some .prim
  | "call" => some .call
  | "batch" => some .batch
  | "scan" => some .scan
  | _ => none

def axisOf (x : Json) : Option (Option Nat) := match x with
  | .null => some none
  | x => (natJ? x).map some

def leavesOf (d : Defn) (outs : Array Ty) : Bool → Nat → Option Nat :=
  fun p i => if p then d.params[i]?.map (·.2.leaves) else outs[i]?.map (·.leaves)

def NodeReads (P : Prog) (types : Array Ty) (d : Defn) (outs : Array Ty) (k : Nat) (nd : Json) : Prop :=
  let node := d.body.nodes[k]!
  let callee := P.defs[node.fn]!
  formOf ((nd.getObjValAs? String "form").toOption.getD "") = some node.form ∧
  (node.out == outs[k]!) = true ∧
  (if node.form == .batch then ∃ xs, nd.getObjVal? "axes" = .ok (.arr xs) ∧ xs.mapM axisOf = some node.axes
    else node.axes = #[]) ∧
  (if node.form == .batch || node.form == .scan then (nd.getObjVal? "n").toOption.bind natJ? = some node.n
    else node.n = 1) ∧
  (if node.form == .scan then (nd.getObjVal? "n_xs").toOption.bind natJ? = some node.nxs else node.nxs = 0) ∧
  (∃ (ajs : Array Json) (raw : Array Arg), nd.getObjVal? "args" = .ok (.arr ajs) ∧ raw.size = ajs.size ∧
    (∀ a < ajs.size,
      (∃ t : Int, intJ? (((ajs[a]!).getObjVal? "type").toOption.getD Json.null) = some t ∧ 0 ≤ t ∧
        t < types.size ∧ (raw[a]!).ty = types[t.toNat]!) ∧
      ∃ qs, parseParts (((ajs[a]!).getObjVal? "refs").toOption.getD Json.null) (leavesOf d outs)
          (fun k2 x => if k2 < k then (d.body.nodes[k2]!).args[x]?.map (·.seq) else none) = .ok qs ∧
        (raw[a]!).seq = Seq.ofParts qs ∧ (raw[a]!).dims = none) ∧
    buildNode d.key k node.form callee raw node.axes node.n node.nxs = .ok (node.out, node.args)) ∧
  d.body.offsets[k + 1]! = d.body.offsets[k]! + (if node.form == .prim then 1 else node.members * callee.gates)

def PrimReads (j : Json) (d : Defn) : Prop :=
  let e := entry j d.key
  d.prim = true ∧ d.body = default ∧ (e.getObjValAs? String "form").toOption.getD "" = "primitive" ∧
  d.id = (e.getObjValAs? String "id").toOption.getD "" ∧
  paramsOf ((e.getObjVal? "params").toOption.getD Json.null) s!"primitive {d.key}" = .ok d.params ∧
  Ty.ofJson ((e.getObjVal? "ret").toOption.getD Json.null) = .ok d.ret ∧ d.binding = (inputId? d.key).map (·.1) ∧
  d.gates = 1 ∧ d.paramLeaves = d.params.foldl (· + ·.2.leaves) 0 ∧ d.outLeaves = 1

def CompReads (j : Json) (P : Prog) (i : Nat) (d : Defn) : Prop :=
  let e := entry j d.key
  let b := (e.getObjVal? "body").toOption.getD Json.null
  d.prim = false ∧ d.binding = none ∧ e.getObjValD "form" = .str "composite" ∧
  d.id = (e.getObjValAs? String "id").toOption.getD "" ∧
  paramsOf ((b.getObjVal? "params").toOption.getD Json.null) s!"definition '{d.key}'" = .ok d.params ∧
  d.body.params = d.params ∧
  Ty.ofJson ((b.getObjVal? "ret_type").toOption.getD Json.null) = .ok d.ret ∧ d.body.retType = d.ret ∧
  d.paramLeaves = d.params.foldl (· + ·.2.leaves) 0 ∧ d.outLeaves = d.ret.leaves ∧ d.gates = d.body.offsets.back! ∧
  ∃ (nds : Array Json) (types outs : Array Ty), b.getObjVal? "nodes" = .ok (.arr nds) ∧ nodesOf e = nds ∧
    (tab j "types").mapM Ty.ofJson = .ok types ∧
    nds.mapM (fun nd => at_ types ((nd.getObjVal? "out").toOption.getD Json.null) "types") = .ok outs ∧
    d.body.nodes.size = nds.size ∧ d.body.offsets.size = nds.size + 1 ∧ d.body.offsets[0]! = 0 ∧
    (∀ k < nds.size, (d.body.nodes[k]!).fn < i ∧
      calleeAt (tab j "callees") nds[k]! = some (.str (P.defs[(d.body.nodes[k]!).fn]!).key) ∧
      NodeReads P types d outs k nds[k]!) ∧
    ∃ qs, parseParts ((b.getObjVal? "ret").toOption.getD Json.null) (leavesOf d outs)
        (fun k2 x => d.body.nodes[k2]?.bind fun y => y.args[x]?.map (·.seq)) = .ok qs ∧
      d.body.ret = Seq.ofParts qs

def DefReads (j : Json) (P : Prog) (i : Nat) (d : Defn) : Prop :=
  P.index[d.key]? = some i ∧ (PrimReads j d ∨ CompReads j P i d)

/-- **The decoder builds each Definition from its entry and its callees.** Stated and not proved: the invariant over
`getDef`, `composite` and `staticFns`. -/
def DecodeReads : Prop :=
  ∀ (text : String) (j : Json) (P : Prog), Flock.Program.decode text.toUTF8 = .ok P → Flock.JsonParse.parse text = .ok j →
    (∀ (c : String) (i : Nat), P.index[c]? = some i → i < P.defs.size ∧ (P.defs[i]!).key = c) ∧
    ∀ i < P.defs.size, DefReads j P i (P.defs[i]!)

/-- **Equivalent canonical entries and the same callees give the same Definition**: two Definitions of one key whose
canonical entries are equivalent and whose nodes' callees of one key are one Definition are one Definition. Stated and
not proved: the canonical entries read field by field (`DigestDecode`, `DigestRefs`, `DigestParts`). -/
def ReadsPin : Prop :=
  ∀ (j j' : Json) (P P' : Prog) (i i' : Nat) (ce ce' : Json), Wf j → Wf j' → i < P.defs.size → i' < P'.defs.size →
    (P.defs[i]!).key = (P'.defs[i']!).key → DefReads j P i (P.defs[i]!) → DefReads j' P' i' (P'.defs[i']!) →
    canonicalEntry (tab j "callees") (tab j "types") (entry j (P.defs[i]!).key) = .ok ce →
    canonicalEntry (tab j' "callees") (tab j' "types") (entry j' (P'.defs[i']!).key) = .ok ce' → JsonEquiv ce ce' →
    (∀ n < (P.defs[i]!).body.nodes.size, n < (P'.defs[i']!).body.nodes.size →
      (P.defs[(P.defs[i]!).body.nodes[n]!.fn]!).key = (P'.defs[(P'.defs[i']!).body.nodes[n]!.fn]!).key →
      Same P P' (P.defs[i]!).body.nodes[n]!.fn (P'.defs[i']!).body.nodes[n]!.fn) →
    Same P P' i i'

theorem reads_node {j : Json} {P : Prog} {i : Nat} {d : Defn} (h : DefReads j P i d) {n : Nat}
    (hn : n < d.body.nodes.size) :
    (entry j d.key).getObjValD "form" = .str "composite" ∧ (d.body.nodes[n]!).fn < i ∧
      ∃ nd ∈ nodesOf (entry j d.key), calleeAt (tab j "callees") nd = some (.str (P.defs[(d.body.nodes[n]!).fn]!).key) := by
  obtain ⟨-, hp | hc⟩ := h
  · obtain ⟨-, hb, -⟩ := hp
    rw [hb] at hn
    exact absurd hn (Nat.not_lt_zero n)
  · obtain ⟨-, -, hf, -, -, -, -, -, -, -, -, nds, types, outs, -, hnds, -, -, hsz, -, -, hk, -⟩ := hc
    rw [hsz] at hn
    obtain ⟨hlt, hcal, -⟩ := hk n hn
    refine ⟨hf, hlt, nds[n]!, ?_, hcal⟩
    rw [hnds, getElem!_pos nds n hn]
    exact Array.getElem_mem hn

theorem entriesPin_of_reads (hD : DecodeReads) (hR : ReadsPin) : EntriesPin := by
  intro text text' j j' P P' key k k' hP hP' hj hj' hA hk hk'
  obtain ⟨hI, hRd⟩ := hD text j P hP hj
  obtain ⟨hI', hRd'⟩ := hD text' j' P' hP' hj'
  have hw := FlockComplete.JsonRT.parse_wf text j hj
  have hw' := FlockComplete.JsonRT.parse_wf text' j' hj'
  suffices H : ∀ i, ∀ (c : String) (i' : Nat), Reaches j key c → Reaches j' key c → P.index[c]? = some i →
      P'.index[c]? = some i' → Same P P' i i' from H k key k' .self .self hk hk'
  intro i
  induction i using Nat.strongRecOn with
  | _ i ih =>
  intro c i' hc hc' hi hi'
  obtain ⟨hlt, hkey⟩ := hI c i hi
  obtain ⟨hlt', hkey'⟩ := hI' c i' hi'
  obtain ⟨ce, ce', hce, hce', he⟩ := (hA c hc).2
  have hr := hRd i hlt
  have hr' := hRd' i' hlt'
  refine hR j j' P P' i i' ce ce' hw hw' hlt hlt' (hkey.trans hkey'.symm) hr hr' (hkey ▸ hce) (hkey' ▸ hce') he ?_
  intro n hn hn' hck
  obtain ⟨hf, hfn, nd, hnd, hcal⟩ := reads_node hr hn
  obtain ⟨hf', hfn', nd', hnd', hcal'⟩ := reads_node hr' hn'
  have hlt2 := Nat.lt_trans hfn hlt
  have hlt2' := Nat.lt_trans hfn' hlt'
  refine ih _ hfn (P.defs[(P.defs[i]!).body.nodes[n]!.fn]!).key _ ?_ ?_ (hRd _ hlt2).1 ?_
  · exact Reaches.callee _ _ nd hc (hkey ▸ hf) (hkey ▸ hnd) hcal
  · refine Reaches.callee _ _ nd' hc' (hkey' ▸ hf') (hkey' ▸ hnd') ?_
    rw [hck]
    exact hcal'
  · rw [hck]
    exact (hRd' _ hlt2').1

theorem digestPins_of_reads (hD : DecodeReads) (hR : ReadsPin) : DigestPins :=
  digestPins_of_entriesPin (entriesPin_of_reads hD hR)

end InferenceOnlyCheck.Properties.Record.SecurityProofs

end
