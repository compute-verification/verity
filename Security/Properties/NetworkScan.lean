import Properties.NetworkScan.Assumptions

/-!
# The rows form's scan fits its layout: what `NetworkCheck.Period.scanOf`'s acceptance gives the sampled audit's scope

The sampled audit's scope at a link-window's statement (`NetTiming.ScopeAcc`) has two halves: the sampling scope
(`Claims.Sampling.Scope`, the sampled verifier's own, which `Claims.Sampling.ScopeMatches` equates with its scope
check), and the statement's layout fitting its program over its horizon (`NetTiming.BoundaryGates.Fits`). The network
check reads the second off the scan's Program (`scanOf`, the generator form); the first it doesn't read.

**The program** is the scan's audit's members' Boolean programs side by side (`jointOf`, in
`Properties.NetworkScan.Flattens`): at the templates' circuit files `fs`, one a template, each laid out by Bits as its
template (`FilesOf`), and the members `ms` over them, each template's instances filling VUs of their own
(`MembersOf`).

**The layout** (`scanLayout`) is the generator form's, at the schedule's `S = lanes · slots` slots of `B` bytes and
`T` buckets, over one egress link, at Bits' positions. At step `t`, slot `σ`'s presence gate is bit 0 of egress instance
`t`'s verdict, its return's field 0, and its frame's bits are bits `8σB … 8(σ + 1)B − 1` of its return's row, field 1
(row bit `8k + i` bit `i` of byte `k`), each its output bit (`SampledProofs.Bits.outBit`). Step `t`'s ingress is bucket
`t`'s `row` parameter, its parameter 0, in bit order: bucket 0's first-template instance, bucket `t ≥ 1`'s
later-template instance `t − 1`, each bit its input bit (`SampledProofs.Bits.inBit`). A field's bits are packed 16 a
port word (`wordBit`), an instance is a copy of a VU of its member (`instPos`), the templates are the root's nodes that
call a composite (`NetworkCheck.Period.templateNodes`, as `scanOf` reads them), and the wires are Bits'
(`SampledProofs.Bits.wiresOf`). A statement's layout is its descriptor's (`layOf`), at horizon `T`.

* `ScanFits`: an accepted scan's layout fits its audit's joint program over `T` steps.
* `ScanScopeAcc`: given an accepted scan, at every deployment whose scope check is its scope, a statement the scope
  check takes, each parse of which has the scan's canonical bytes as its descriptor and the audit's joint program as
  its circuit, is in the sampled audit's scope at `layOf` and `T` (`ScopeAcc`).
-/

namespace Properties.NetworkScan

open NetworkCheck.Period
open SampledProofs.Bits (Members inBit outBit wiresOf)
open NetworkCheck.Circuit (templateDef fieldsOf)

/-- **The scan's layout** in its Program `P` at the schedule `s`, its templates' circuit files `fs` and its audit's
members `ms`, over one egress link: step `t`'s slot `σ` is bit 0 of egress instance `t`'s verdict and the `8B` bits of
its frame in the instance's returned row, step `t`'s ingress is bucket `t`'s `row` parameter, and the wires are Bits'. -/
def scanLayout (P : Flock.Program.Prog) (s : Schedule) (fs : List Flock.Circuit) (ms : Members) :
    NetTiming.BoundaryGates 1 s.allocation :=
  let ts := templateNodes P
  let e := ts.length - 1
  let rets := fieldsOf (templateDef P (ts.getD e 0)).ret
  let bits := 8 * s.cellBytes
  let out := fun (t π β : ℕ) =>
    instPos P fs ms e t fun c n u k => outBit c n u k (wordBit rets π β).1 (wordBit rets π β).2
  { egress := fun t _ σ => (out t 0 0, (List.range bits).map fun b => out t 1 (σ.val * bits + b))
    ingress := fun t =>
      let it := if t = 0 then (0, 0) else (1, t - 1)
      let ps := (templateDef P (ts.getD it.1 0)).params.toList.map (·.2)
      (List.range (s.allocation * bits)).map fun ℓ =>
        instPos P fs ms it.1 it.2 fun c n u k => inBit c n u k (wordBit ps 0 ℓ).1 (wordBit ps 0 ℓ).2
    wires := wiresOf ms }

/-- A statement's layout at the schedule `s`, the files `fs` and the members `ms`: its descriptor's scan layout, none
where the descriptor doesn't decode. -/
def layOf (s : Schedule) (fs : List Flock.Circuit) (ms : Members) (x : Claims.Sampling.AuditStatement) :
    NetTiming.BoundaryGates 1 s.allocation :=
  match Flock.Program.decode ⟨x.desc.toArray⟩ with
  | .ok P => scanLayout P s fs ms
  | .error _ => { egress := fun _ _ _ => (0, []), ingress := fun _ => [] }

/-- **An accepted scan's layout fits its audit's joint program.** If `scanOf` accepts the scan `j` as the registered
Program `program` at the schedule `s`, its canonical text is `c` and decodes to `P`, `fs` are `P`'s templates' circuit
files, each laid out by Bits as its template (`FilesOf`), and `ms` its audit's members over them (`MembersOf`), then
`P`'s scan layout fits the members' joint program (`jointOf`) over the window's `T` steps: every egress gate is returned
and not an input, every ingress gate is an input, no egress gate of a step depends on an ingress gate of a later step
through reads and wires, and the wiring fits. -/
def ScanFits : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (fs : List Flock.Circuit) (ms : Members),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → FilesOf P fs →
      MembersOf P fs ms → (scanLayout P s fs ms).Fits (jointOf ms) s.window

/-- **An accepted scan's statements are in the sampled audit's scope.** If `scanOf` accepts the scan `j` at the
schedule `s`, its canonical text is `c` and decodes to `P`, `fs` are `P`'s templates' circuit files (`FilesOf`) and
`ms` its audit's members over them (`MembersOf`), then at every deployment `D` whose scope check is its sampling scope
(`ScopeMatches`), a statement `x` the scope check takes, every parse of which has `c`'s bytes as its descriptor and
the members' joint program as its circuit (`Flattens`), is in `ScopeAcc` at the layout `layOf s fs ms` and the horizon
`T`. -/
def ScanScopeAcc : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String) (P : Flock.Program.Prog)
    (fs : List Flock.Circuit) (ms : Members),
    scanOf j program s = .ok sc → Flock.canon j = .ok c → Flock.Program.decode c.toUTF8 = .ok P → FilesOf P fs →
    MembersOf P fs ms →
    ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      (D : Claims.Sampling.Deploy Coin Col T) (x : Claims.Msg),
      ScopeMatches D → D.scopeCheck x = true →
      (∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8 ∧ Flattens ms st.prog) →
      NetTiming.ScopeAcc D (layOf s fs ms) (fun _ => s.window) x

end Properties.NetworkScan
