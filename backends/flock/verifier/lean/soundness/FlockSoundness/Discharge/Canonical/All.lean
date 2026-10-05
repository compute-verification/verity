import FlockSoundness.Discharge.Canonical.Headline
import FlockSoundness.Discharge.Canonical.Statements
import FlockSoundness.Discharge.Canonical.PubIn
import FlockSoundness.Discharge.Canonical.Gaps

/-!
# The canonical I/O format's binding

`PROTOCOL.md` §16.16's format, and the binding statements and session headline at it:

* `Format`: `formatOk`, the format as one check on the parsed circuit, and `formatOk_spec`;
* `Chain`, `Commit`: a row of any length, its last block partial and in the `hm96` slot (`row_commitC`, `row_bc_C`,
  `bc_site_commitC`);
* `Shared`: the registered `b ‖ c` with or without shared rows (`bc_site_ofC`, proved);
* `Reads`, `Outputs`: unit inputs as row bits (`keysRowC_of_parse`), output rows of runs of output ports
  (`outBitRowC_ne_none`, `out_copyC`);
* `Sites`, `Rows`, `Session`: the sites, `lay` and `rs` at `sha512/row/v2` widths, and `zk_session_soundC`;
* `Verifier`: `parse_formatOk`, what verifier change V1's statement check gives (`ProvedScope.canonicalOk`);
* `Headline`: `zk_session_sound_canonical`, over what flock-verify accepts;
* `Statements`: `RowsCommittedC`, `OutputsCommittedC`, `OutputsBindingC` at the statement level, tails included;
* `Gaps`: what the headline does not yet cover. Tails' outputs (`OutputsComputedC`, after V2), public inputs
  (`PublicPinnedC`) and registered reads (`RegisteredRowC`, after V3).

Every declaration is proved but three, all `sorry` and all in `Gaps`: `parse_outputsComputed` (V2), `public_pinnedC`
and `registered_rowC`. So `FlockSoundness.lean` imports every module here but `Gaps` and this one, since `check`'s
audit admits no `sorry`.
-/
