[2026-11-11T13:17Z] 7d38fafe7e0f
WIRED to @training, (iii) firmed. Everything below is READ at 195facd5dd9b (my branch head; the files named are
byte-identical at FORM's 78a418d0b3d6 and at origin/main 967b722429 except Record.lean, which this does not cite).
Nothing was run but `rg` and `sed`. No experiment, no code.

RESULT: I withdraw the sentence as I sent it. "A developer who registers a Program that complies beside a circuit
file that lowers another one passes both" is wrong about WHO: the developer registers neither. Both are the
verifier's own parameters, and the tree says the auditor builds the circuit file. What is true, and unchecked, is
about the auditor's lowering, not about a cheating prover. Exact version at the end.

## The two files, and which check accepts each

- The Program: a `verity-ir/descriptor/v1` descriptor, as text. It is the `program` key of the inference-only check's
  params, "the verifier's own copies" (`NciCheck/Protocol.lean` 24-28), and the `program` key of sampled proofs'
  params, "the verifier's own copies ... `program` (its descriptor)" (`Spot/OneStage.lean` 4).
  The inference-only check (`NciCheck.Protocol.check`, Protocol.lean 146-153) reads the Program, the record and the
  evidence's `owner`, `theta`, `provenance`. It reads no circuit file and no blob: `rg -n "circuit|blob"` over
  Protocol.lean finds only "the declared circuit", which the header defines as the Program (line 27).
- The circuit file: flat hm-out text, named by its SHA-512 in sampled proofs' params, `proofs.circuit`: "the SHA-512s
  the auditor committed for the circuit and public files" (`Spot/OneStageZk.lean` 5-7; read at 178-181; the blob is
  fetched under that digest at 580). Sampled proofs v2 runs Flock's verifier on that file's rows
  (`Flock.ProvedScope.circuitOf tags I.circuitFile I.tables`, OneStageZk.lean 589-592). The composite's sampled
  component is v2 only (`VStarRecursive/Sampled.lean` 2-5).
- Who writes the circuit file: `spot/verifier/one_stage/proofs.py` 8-11: `verify` "refuses a circuit file that doesn't
  hash to the committed digest before anything runs, so each verdict is over the circuit the auditor's own lowering
  built. The circuit file's tie to the program's Definitions rests on that lowering: the verifier holds the file to
  the program only as far as its META names it, until it derives the circuit from the descriptor." Line 411: "prove on
  the circuit file the auditor built". PoUW says the same of its files (`PouwCheck/CommittedFiles.lean` 27-33).
- In the composite's game both are fixed before the play: `Target` is "for all θ, F, c, σ" (`Game.lean` 890-892), θ
  the parameters, and the circuit file is "the committed file the parameters name by the circuit file's digest"
  (`Vocab.circuitFile`, Game.lean 738-739). The strategy σ chooses neither.

## Every tie between them that is checked today

1. Inference-only's Program is the record's: `names` (Protocol.lean 138-141): the record's `program` is
   hex(`P.sha512`), SHA-512 of the canonical descriptor without `annotations` (`Flock/Program.lean` 358-359).
2. The policy and sampled proofs commit to ONE record: `Core.Remote.tied` (`CoreRemote.lean` 52), which `SameAccepts`
   reads as `OneRecord` (`Game.lean` 228-231; `security/Properties/Composition.lean` 77-83). This is the only tie
   `SameAccepts` makes between the two components. It ties their commitments. It ties no params: `Accepted` gives each
   component the params the composition names for it (Composition.lean 62-66) and compares none. The other tie,
   `Remote.windows`, is PoUW's windows on the policy's record (CoreRemote.lean 67-68).
3. Sampled proofs' Program is the record's: R2 (`Spot/Registration.lean` 491-501): the partition object's `program`
   and the record's `program` are hex of its own Program's SHA-512. With 1 and 2: both components hold the same
   Program, up to a SHA-512 collision.
4. The circuit file names that Program: `checkPartition` (`Flock/HmRow.lean` 1288-1303): META `partition.digest` is
   the digest of the verifier's own partition object, and that object's `program` is META `program_sha512`. An
   accepted session counts only when the verifier's own Program's digest is the partition object's `program`
   (`push`, OneStageZk.lean 531-532).
5. The circuit file names a template of that Program: `deriveTemplateUnits` (HmRow.lean 1197-1218): the verifier's
   Program hashes to the partition object's `program`; META `program_digests`' one key is a template the query lists;
   every unit the statement claims is an instance of that template in the verifier's own Program.
6. The file's units are the Program's population in order: `FilePart.check` (OneStageZk.lean 593; `Spot/FilePart.lean`
   15-20).
7. Registered reads: the template's parameters are held to the verifier's registered file (`Flock/Registered.lean`
   462-474).

Every one of 4 to 7 reads the file's META, header or unit count. None reads the file's rows against the Definition's
body. The verifier says so itself for a size class's circuit: "That the class circuit lowers that Definition is the
class's own claim, which the verifier does not check" (HmRow.lean 1184-1187), and Flock's ledger for every file: "the
lowering does no optimizing. So no hypothesis relates the rows to another circuit" (`Proofs/Flock/Soundness/
ASSUMPTIONS.md` 16-18).

## What is left unchecked, in one sentence

That the gates written in the circuit file compute the Definition the file's header names: the auditor's lowering
(Python, `verity_flock.ir_lower` and `verity_flock.circuit.compose`, circuit.py 682) writes them, the verifier checks
the header's names and digests against the Program and then proves the rows as they stand, and no Lean reads the
rows against the Definition.

## So, for a cheating prover

- With the auditor building the file (what the tree states): the prover computes what it likes and gets no verdict it
  should not from this. It cannot swap the file: `proofs.circuit` is the auditor's digest, and a second file with
  that digest is a collision (`Vocab.fileFinder`, Game.lean 823-824). The exposure is a bug in the auditor's
  lowering, which is trusted code that nothing checks: a lowering that drops or changes a gate gives an accepted
  proof of another computation with every name in place.
- Only if a deployment let the developer supply the circuit file or its digest would my sentence hold as sent: the
  developer lowers Program Q, writes Program P's digest and template name into META, registers P; the policy accepts P
  and sampled proofs accepts Q's rows. I found no place in the tree where the developer supplies it, and I did not
  read the demos' or the adapter's launch code for it.
- For `InferenceOnlySound` this means: the theorem is over all θ, so a proof of it on the file's circuit needs either
  a premise on θ (the named circuit file is the lowering of the named Program; a premise on the auditor's own tool),
  or the verifier deriving the circuit from the descriptor, which is what proofs.py 11 says is coming. It does not
  need a new check on the prover.

## What this does not touch

"False as stated" stands, and it is independent of who lowers: with an honest lowering the file's circuit is still n
copies of one unit in bits, with no gate reading another unit (`Spot/FilePart.lean` 5-9: "No unit reads another's
sources"; Keyed.lean 33-39). The wires between units are in the record, not in the file. And the check's own header
calls the Program "the declared circuit" (Protocol.lean 11, 27), where the game's `declaration` is the file's circuit
(Game.lean 755-756): the check and the game's claim name two circuits today.

NOT READ: `NciCheck/Record.lean`, `Flock/Registered.lean` past 462-474, `Spot/Registration.lean` outside R2,
`verity_flock/ir_lower.py`, the demos' and the adapter's launch code, onsite's composite (`CoreOnsite.lean`).
The rest (readings A and B, the design rule) follows by Thu noon PST.

