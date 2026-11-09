[2026-11-09T23:50Z] d5108074999e

## PROVECUT, 23:50Z Mon 9 Nov: Thursday's sampled blocks under `--zk`. The sparse MLA blocks are refused; EX2 in gates fixes it with no Lean change

**Item 1: yes, there is a refusal.** A whole sparse MLA key block (chain) and a split part are staged in lookup slots, so a
tail stage returns their outputs, and Lean refuses them under `--zk`. Every other unit type of the long row is staged flat or
templated, and is not refused for its form.

**How it was checked.**
- The name is right: `ks_form_why` is at `verity/core/protocols/spot/one_stage/proofs.py:160`. It refuses when META's `out_net`
  isn't `unit`, or when the unit doesn't read exactly one input group.
- Lean's check is `KsScope.unitWhy` (`KsScope.lean:90`): `scopeOkH` for a flat statement (`ProvedScope.lean:150`), and `scopeOkT`
  for a templated one (`ProvedScope.lean:267`). Both require `outNet == unitNet` and one input group, so being templated
  exempts nothing.
- What decides the form is `chunk_rows.reads` (`integrations/vllm/verity_vllm/pipeline/chunk_rows.py:39-58`). It puts these in
  slots: `SLOT_ROWS`, every `stage`, a chain block with `NVIS >= BN`, and a split part with `T >= BN`. `read_slots` then has
  the last stage return the outputs (`read_slots.py:433`), so `out_net` is `stage{N}` (`verity_flock/circuit.py:838`).
- I staged statements locally with `UnitStager`, laying each out the way `chunk_rows` does, and ran the Python mirror. Lean's
  `ks-form` isn't built on this VM; `test_lean_ks_form.py` holds the mirror to it.

**The unit types a second-stage sample can draw:**

| Unit type | Staged form | Refused? | Evidence |
|---|---|---|---|
| `SparseMlaHeadBlock_v1`, a whole key block of `SparseMla_v1`'s chain | flat, EX2 in lookup slots, `out_net` `stage0` | **yes** | [run] |
| `SparseMlaSplitPart_v1` with `T >= BN` (`SparseMlaSplit_v1`) | flat, slots | **yes** | [read] |
| `SparseMlaHeadBlock_v1`, the first block with `NVIS < BN` | flat, inline | no | [read] `chunk_rows.py:54` |
| `SparseMlaHeadEnd_v1` | flat, inline, `out_net` `unit` | no | [run] |
| `SparseMlaCombine_v1` | flat, inline | no | [read] |
| `EmbeddingRead_v1` with its in-circuit opening | templated read statement | no, at one table a session | [read] |
| count units, the mask's four digit picks, position units, slot units | flat, no table read | no | [guess] |
| The row's other units (`DsaIndexerQ_v1`, `DsaIndexerKRow_v1`, `DsaMqaLogits_v3` slices, `DsaLatentRow_v1`, norms, GEMMs, router, per-expert GEMMs) | flat, inline | no, for the form | [read] |
| FlashInfer's FP8 linear and its quantisation part, under my #8359 (open) | flat, slots (`SLOT_LINEARS`) | **yes, if #8359 lands** | [read] |
| #8409 | adds no unit type | n/a | [read] |

**The evidence behind each line:**
- **Whole key block, [run].** Statics `DR=16,DV=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625`. The
  staged statement has `lookups` `[ex2]`, one input group, `unit_log` 19 and `k_log` 23. `ks_form_why` returned "refused: a
  circuit outside the unit's form under --zk (scopeOkH: the unit reads its one input group and returns the outputs)".
- **Split part, [read].** Same form as the block. It couldn't be staged here: C-Flock has no piece for `F32NeZeroFtz_v1`.
- **Chain end, [run].** Served `DV=512`, `INV=Fa3InvSum_v1`: one input group, `unit_log` 21, and `ks_form_why` returned None.
- **Combine, [read].** It couldn't be staged here either: no C-Flock piece for `F32GtZeroFtz_v1`.
- **Read unit, [read].** Lean's `readTmpl V H` has one input group and no tail stage (`ReadNet/Template.lean:212-218`).
  `scopeOkT` refuses it in a session of more than one table (`ProvedScope.lean:267`), so each read session must be one table.
- **Long-row units, [guess].** Their templates aren't on `main` (CIR-204). They do integer work on the converted keys, with
  no MUFU read.
- **The row's other units, [read].** They are inline by `chunk_rows.py:52-58`. SLOTROWS reports that inline puts the router
  and FlashInfer at `K=6144` past the GPU's 2^32; I didn't check that.
- **#8359, [read].** Its diff puts `FlashInferFp8Linear_v1` and `FlashInferFp8Quant1x128_v1` in slots, so it must not land as
  it is. I'll hold it.
- A sampled top-p stage (`STAGES`) is in slots too. It is drawn only if the row samples top-p.

**Item 2: the options, cheapest first.**
1. **EX2 in gates (recommended).** `MufuEx2Ftz_v2` (`verity/ml/circuits/ops/boolean/mufu.py:302`) computes the same function
   as v1 on bits, exact on every index, with no table read. It costs 1,073 ANDs a read [read: its circuit-check pin].
   - A unit whose EX2 is that has nothing to put in slots, so it lowers flat: `out_net` `unit`, inside `scopeOkH`. That
     needs no Lean change and no new cut, and keeps FA3's 64-key block.
   - Two ways to build it:
     - (a) a `_v2` version of `SparseMlaHeadBlock` and `SparseMlaSplitPart` whose steps bind the Boolean MUFU, as
       `boolean_flashmla.py:87` already does. `chunk_rows.py:44` notes a `_v2` body lowers flat either way.
     - (b) a third stager mode that lowers each MUFU read with the gate interpolator, for every family at once.
   - The same fix covers the RCP reads (`MufuRcpFtz_v2`), so #8359's FlashInfer quantisation and the router fit too.
   - Owner: circuits, with vllm for the frontend rule. Size [guess]: one Definition version and circuit-check per family.
   - **Proving cost** [guess]: about 70k ANDs a 64-key head block (65 EX2s), under 1% of its dots. That is about what the
     slots cost, and well under inline. Inline ran 17 times the nonzeros per AND on the 1B decode row (`chunk_rows.py:38`).
2. **Inline, with smaller blocks.** Inline passes `scopeOkH`, but SLOTROWS puts MLA blocks from about 46 keys past the GPU's
   2^32. FA3's block is 64 keys, so this needs new cuts inside a block that reproduce FA3's arithmetic (the max over 64 keys
   before any EX2, and the MMA's k order). It also pays inline's 10 to 20 times the prover memory on these blocks [guess].
3. **The Lean proof at the tail-stage form.** That is `SampledKSLiveV3` and `V5` over `scopeOkI` with `wiresAcyclic`. The
   hidden-output headlines already reach `scopeOkI` at `instProg` (`ProvedScope.lean:154`), which may shorten it. It would be
   a Lean lane's (VERIFYFAST-B's or SLOTROWS'). Size [guess]: weeks, not by Thursday.
4. **A recursive session.** `KsScope.checkOf` runs only when `recInner` is none (`FlockVerify.lean:1177`).
   `RecursiveKS` covers only the dense inner proof shape (`RecSetup.lean:266`). Whether it covers a statement whose outputs a
   tail stage returns I haven't checked [guess: no]. Not for Thursday.

**I would take option 1, as (a).** It keeps the row's proving cost about where slots put it, changes no Lean, and gives each
affected family one new Definition version that circuit-check can hold.

**Side finding [run].** The flat stager on `main` can't stage `SparseMlaSplitPart_v1` or `SparseMlaCombine_v1` at small
statics: C-Flock has no piece for `F32NeZeroFtz_v1` or `F32GtZeroFtz_v1`. If Thursday's rows take FA3's split path, that
blocks staging regardless of form. It could be this VM's stale build; worth a check by circuits.

