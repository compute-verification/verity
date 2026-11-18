@circuits @proofs @lean: round 1 of the served classes, at `WindowClass(4,27,2)`, through both readers (lean/WIRED)

| class | Lean, grid `60f93bbc9c1` | Lean, guards `dd9ef367816` | Lean, interfaces `a1c0c40e270` | Python, interfaces | Python, guards |
|---|---|---|---|---|---|
| tied | `program`: decode `E_REINTERPRET unchecked`, 21 s, 7.0 GB | same, 21 s | `program`: grid part not read, node 2092, 13 s | Program accepted, 45 s, 4.7 GB; edges `edge-fields`, node 2090 | same |
| untied | `program`: decode `E_REINTERPRET unchecked`, 25 s, 7.0 GB | same, 23 s | `program`: grid part not read, node 109, 22 s | Program accepted, 60 s, 4.8 GB; edges `edge-fields`, node 107 | same |
| int8 | `program`: decode `E_REINTERPRET unchecked`, 28 s, 7.2 GB | same, 21 s | `program`: grid part not read, node 174, 22 s | Program accepted, 57 s, 4.9 GB; edges `edge-fields`, node 171 | same |

No class passes either reader. All three Programs were built with circuits' command of 1:07, verbatim, on `interfaces` `f55b8e31f66`. Between that commit and `a1c0c40e270`, no file under the builder, `verity/circuits`, `verity/commitments` or the inference-only verifier changed. The Programs are kept in the build runs' `out/` for later rounds: tied `r20261118-092027-fb92`, untied `r20261118-092049-2abf`, int8 `r20261118-092056-98d2`.

The Lean record check ran on a stand-in case: the real Program, `record` `"{}"`, and an empty trusted log. Nothing past the `program` stage reads the record, so every Lean refusal above is real. Circuits' command makes no record or log, so no class can be taken past `program` with it.

**Lean, grid and guards: whose it is.** It is the Lean decoder's, in `verity/circuits/lean/Circuits/Program.lean`, `Widths.partOk`, its `.grid` case. Python accepts the same Program.
- **The full message.** The decoder refuses `definition 'Root[WiredPopulations_v1{…}]_v1': E_REINTERPRET unchecked: its references are read leaf by leaf past 1048576 leaves to compare their widths, past which this verifier refuses them`. Run `r20261118-093136-5347` (grid) and `r20261118-093716-280a` (guards) print it from a scratch probe of `Flock.Program.decode`. `decodeProgram` maps it to the bare `program`.
- **Why it walks.** The `.grid` fast path needs the reading type's one width `w` and, for each lane, `rangeWidths S lo (hi+1) == [w]` over the lane's extremes (`Lane.bounds`). In all three classes the reading type has one width, but each lane's extremes span a writer's mixed-width output. So the decoder walks `q.count` leaves, each a `step` of `WALK = 2^20`.
- **Tied and untied:** the `batch SelectCollect` arguments, of type `Array<4,Array<8016,Value<16>>>` and its `Value<32>` twin, for example `{"grid":[167,[["n",5413,0,1,[[4,334]]],…]]}`. These are 48 lanes of runs of 167 over a node whose 16- and 32-bit fields alternate every 167 leaves. Each argument walks 32,064 leaves. The budget is crossed at root node 7231 (tied) and 8367 (untied).
- **int8:** the `batch ScaledMmS8` argument 0, for example node 174 `{"grid":[2048,[["n",173,0,1,[[96,0],[4,2049]]]]]}`. Each run of 2048 leaves is all `Value<8>`, and the stride 2049 steps over a 32-bit scale. It is crossed at root node 180, with about 6.2·10⁹ leaves in all. This is a replay in Python of the `.grid` case over the root (`r20261118-093641-ee9d`). It counts only the root's grids, so the true crossing is at or before these nodes.
- **The smallest mend, for lean/EDGES** (text only, nothing edited):
  - **Where:** in the `.grid` case of `partOk`, after `if lanes.all (laneOk w) then return true`, under the same `if let some w := ht`.
  - **What:** test lane by lane, each distinct run once, before the leaf walk. Levels of step 0 repeat the same run, so they are skipped. Each run charges one `step`:
    ```lean
    def Lane.runStarts (l : Lane) : List Int :=
      l.lv.foldl (fun acc (r, d) => if d == 0 then acc else acc.flatMap fun s => (List.range r).map fun m => s + m * d) [l.b]
    -- in partOk's .grid case, under `if let some w := ht`:
      let runOk := fun (l : Lane) => do
        match s.ty? l.p l.i with
        | none => return true
        | some S =>
          if s.homo? l.p l.i == some w then return true
          for st in l.runStarts do
            step
            let e := st + ((run : Int) - 1) * l.e
            if rangeWidths S (min st e).toNat ((max st e).toNat + 1) != [w] then return false
          return true
      if ← lanes.allM runOk then return true
    ```
  - **Why it is sound:** it accepts only where every leaf between a run's extremes has width `w`. Otherwise it falls through to today's leaf walk.
  - **Its cost on these Programs:** 4 steps per lane per part (192 per `SelectCollect` part, 4 per `ScaledMmS8` part), against today's 32,064 and 786,432 to 8,388,608 leaves. One step per lane per period, as Python's `Grid.pieces` reads it, would still be 1,673,728 steps for int8, over budget. That is why the step-0 levels must be collapsed.
  - **Its proof:** `security/Proofs/Properties/InferenceOnly/LeafWidthsPart.lean`'s `grid` case, at line 318, needs one more branch: leaf `k`'s reference lies in the run of its lane starting at a member of `runStarts`.
- **Lean, interfaces** has no grid part, so it refuses at the first one, as before.

**Python edges reader: `edge-fields` in all three, on both branches.** `verity.commitments.periods.edges._part_wires` refuses at `_crosses_fields`, where a part's rows read more than one output field of one writer.
- **tied:** root node 2090, `RoPEHead_v1{D=64}`, argument 0 port `x` (64 leaves, uncut, 128 members). The part is `["g",2048,[[false,1642,0,1,[[4,3072]]]],8192,false]`. The writer is node 1642, `LinearOut_v3{N=3072,W=64}`, a batch of 4 whose output is `Tuple` of 48 `Array<64,Value<16>>` fields. Each RoPE member reads one whole field, but one part spans 32 fields of each writer member. A wire names one `o<f>`, so it is refused.
- **untied:** root node 107, `EmbeddingRead_v2{V=128256,H=2048}`, port `tok` (1 leaf, 4 members), part `["g",4,[[false,106,0,160,[]]],4,false]`.
- **int8:** root node 171, `EmbeddingRead_v2{V=128256,H=2048}`, port `tok`, part `["g",4,[[false,170,0,160,[]]],4,false]`. The stride 160 reads a different field of the writer for each member.
- **Whose it is.** By the code, the builder's parts step across a writer's fields, which the edges rule refuses by name. The mend is either the builder's (lay a writer's output out as one `Array` field, so a step indexes it) or proofs' (split such a part into one wire per writer field). Circuits and proofs decide which.
- **The twin's `own_rows`**, run form only and without a record, also refuses `own-rows` in all three. That is the next stage after the record. It is circuits' known 106 parts sharing a registered leaf, not re-derived here.

Runs:
- Builds: `fb92`, `2abf`, `98d2`.
- Lean builds: `0b8e`, `32e5`, and guards `5175` (the first, `6a4b`, failed writing `Spot/Bits.olean` and was rerun).
- Lean probes: `5347`, `54e9`, `280a`.
- Python readers: `7cb6`, `ce8b`, `7c68`, `2cc7`, `e5d6`, `18fc`.
- Grid replay and edges detail: `ee9d`.

Each run's id is `r20261118-…-<id>`. Next round: about 2:15. I rebuild only if circuits' builder files moved.
