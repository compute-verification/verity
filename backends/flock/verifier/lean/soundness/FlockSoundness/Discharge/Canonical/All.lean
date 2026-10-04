import FlockSoundness.Discharge.Canonical.Headline
import FlockSoundness.Discharge.Canonical.Statements

/-!
# The canonical I/O format's binding, frozen

`PROTOCOL.md` §16.15's format, and the binding statements and session headline at it:

* `Format`: `formatOk`, the format as one check on the parsed circuit, and `formatOk_spec`;
* `Chain`, `Commit`: a row of any length, its last block partial and in the `hm96` slot (`row_commitC`, `row_bc_C`,
  `bc_site_commitC`);
* `Shared`: the registered `b ‖ c` with or without shared rows (`bc_site_ofC`, proved);
* `Reads`, `Outputs`: unit inputs as row bits (`keysRowC_of_parse`), output rows of runs of output ports
  (`outBitRowC_ne_none`, `out_copyC`);
* `Sites`, `Rows`, `Session`: the sites, `lay` and `rs` at `sha512/row/v2` widths, and `zk_session_soundC`;
* `Verifier`: `parse_formatOk`, the verifier change V1 as a statement (false at main);
* `Headline`: `zk_session_sound_canonical`, with no scope hypothesis;
* `Statements`: `RowsCommittedC`, `OutputsCommittedC`, `OutputsBindingC` at the statement level, tails included.

Not imported by `FlockSoundness.lean`: the stubs are `sorry`s, and `check`'s audit admits none.
-/
